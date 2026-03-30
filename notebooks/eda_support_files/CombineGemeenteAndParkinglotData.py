from __future__ import annotations

import logging
from typing import Tuple

import geopandas as gpd
import pandas as pd


class CombineGemeenteAndParkinglotData:
    """
    Combine parking-lot data with municipality (gemeente) polygons.

    Workflow
    --------
    1. Assign each parking lot to a gemeente using a spatial join.
    2. If a parking lot intersects multiple gemeenten, pick the one with the
       largest intersection area.
    3. Count the number of parking lots per gemeente.
    4. Evaluate whether each gemeente has enough parking lots relative to its
       required number (`Benodigd_ceiling`).

    Parameters
    ----------
    logger : logging.Logger | None
        Optional injected logger. Defaults to named logger for this class.
    """

    def __init__(self, logger: logging.Logger = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        gdf_parking_lots: gpd.GeoDataFrame,
        gdf_gemeenten: gpd.GeoDataFrame,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Execute the full municipality-parking assignment + ratio-check pipeline.

        Parameters
        ----------
        gdf_parking_lots : GeoDataFrame
            Parking lot geometries.
        gdf_gemeenten : GeoDataFrame
            Municipality polygons (must contain 'gemeentenaam' and 'Benodigd_ceiling').

        Returns
        -------
        (gdf_parking_lots, gdf_gemeenten) : tuple of GeoDataFrame
            Updated parking lot dataframe (with gemeente assigned)
            and updated gemeente dataframe (with counts & ratios).
        """
        if "gemeentenaam" not in gdf_parking_lots.columns:
            self.logger.info("Assigning gemeenten to parking lots via spatial join.")
            joined = self._join_parking_to_gemeenten(gdf_parking_lots, gdf_gemeenten)
            gdf_parking_lots = self._resolve_multiple_gemeenten(joined, gdf_gemeenten)

        self.logger.info("Calculating parking-lot availability per gemeente.")
        gdf_gemeenten = self._calc_gemeente_ratios(gdf_parking_lots, gdf_gemeenten)

        return gdf_parking_lots, gdf_gemeenten

    # ------------------------------------------------------------------
    # Spatial Join
    # ------------------------------------------------------------------
    def _join_parking_to_gemeenten(
        self,
        gdf_parking_lots: gpd.GeoDataFrame,
        gdf_gemeenten: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """
        Spatial join linking each parking lot to intersecting gemeenten.
        Some parking lots may join to multiple municipalities.
        """
        if "geometry" not in gdf_parking_lots or "geometry" not in gdf_gemeenten:
            raise ValueError("Both input GeoDataFrames must contain a 'geometry' column.")

        self.logger.info("Performing spatial join (predicate='intersects').")

        joined = gpd.sjoin(
            gdf_parking_lots,
            gdf_gemeenten[["gemeentenaam", "geometry"]],
            how="left",
            predicate="intersects",
        )

        before = len(joined)
        joined = joined[~joined["gemeentenaam"].isna()].copy()
        after = len(joined)

        self.logger.info(
            "Removed %d parking lots outside the municipality bounds.",
            before - after,
        )
        return joined

    # ------------------------------------------------------------------
    # Resolve parking lots intersecting multiple gemeenten
    # ------------------------------------------------------------------
    def _resolve_multiple_gemeenten(
        self,
        joined: gpd.GeoDataFrame,
        gdf_gemeenten: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """
        When a parking lot intersects multiple municipalities, select the
        one with the largest geometric intersection area.
        """

        if "index_right" not in joined.columns:
            raise ValueError("Spatial join missing 'index_right'; cannot resolve matches.")

        self.logger.info("Resolving multiple gemeente matches per parking lot.")

        # Compute intersection area for each (parking lot, gemeente) candidate.
        joined = joined.copy()

        joined["intersection_area"] = joined.apply(
            lambda row: row.geometry.intersection(
                gdf_gemeenten.loc[row.index_right, "geometry"]
            ).area
            if row.index_right is not None
            else 0.0,
            axis=1,
        )

        # The groupby key is the parking-lot original index
        gemeente_col = (
            joined.groupby(joined.index)
            .apply(self._pick_largest_area)
            .rename("gemeentenaam")
        )

        joined["gemeentenaam"] = gemeente_col
        joined = joined.drop(columns=["index_right", "intersection_area"])

        return joined

    @staticmethod
    def _pick_largest_area(group: gpd.GeoDataFrame) -> str:
        """Pick the gemeente with the largest intersection area for a parking lot."""
        if len(group) == 0:
            raise ValueError("Group is empty; cannot pick largest area.")
        idx = group["intersection_area"].idxmax()
        return str(group.loc[idx, "gemeentenaam"])

    # ------------------------------------------------------------------
    # Counting & Ratios
    # ------------------------------------------------------------------
    def _calc_gemeente_ratios(
        self,
        gdf_parking_lots: gpd.GeoDataFrame,
        gdf_gemeenten: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """
        Count parking lots per gemeente and evaluate:
            genoeg_parkeerplaatsen = aantal_parkeerplaatsen >= Benodigd_ceiling
            ratio_parkeerplaatsen  = aantal_parkeerplaatsen / Benodigd_ceiling

        Raises ValueError if any gemeente does not meet its required threshold.
        """

        if "Benodigd_ceiling" not in gdf_gemeenten.columns:
            raise ValueError("Column 'Benodigd_ceiling' missing in gemeente dataframe.")

        counts = (
            gdf_parking_lots.groupby("gemeentenaam")
            .size()
            .rename("aantal_parkeerplaatsen")
        )

        # Attach counts to gemeenten
        gdf_gemeenten = gdf_gemeenten.copy()
        gdf_gemeenten["aantal_parkeerplaatsen"] = gdf_gemeenten["gemeentenaam"].map(counts).fillna(0).astype(int)

        # Compute ratios
        gdf_gemeenten["genoeg_parkeerplaatsen"] = (
            gdf_gemeenten["aantal_parkeerplaatsen"]
            >= gdf_gemeenten["Benodigd_ceiling"]
        )

        gdf_gemeenten["ratio_parkeerplaatsen"] = (
            gdf_gemeenten["aantal_parkeerplaatsen"]
            / gdf_gemeenten["Benodigd_ceiling"]
        )

        # Validate: ensure all gemeenten have enough parking lots
        if not gdf_gemeenten["genoeg_parkeerplaatsen"].all():
            missing = gdf_gemeenten.loc[
                ~gdf_gemeenten["genoeg_parkeerplaatsen"], "gemeentenaam"
            ].tolist()

            raise ValueError(
                "Not enough parking lots for the following gemeenten: "
                + ", ".join(missing)
            )

        return gdf_gemeenten
