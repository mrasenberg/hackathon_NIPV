from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Union

import geopandas as gpd
import pandas as pd


class FilterOSMData:
    """
    Filter OSM parking polygons by (a) allowed parking types, (b) minimum size,
    and (c) overlapping geometries (keep the largest polygon).

    Workflow
    --------
    1) Optionally load a preprocessed GeoJSON from disk.
    2) Filter by allowed parking types.
    3) Ensure/compute 'area_m2' (in meters) and filter by minimum size.
    4) Remove polygons fully covered by larger polygons.
    5) Save to GeoJSON to avoid reprocessing.

    Parameters
    ----------
    allowed_parking_types : Iterable[str | None], optional
        Whitelist of acceptable values in the 'parking' column.
    min_parking_size : float, optional
        Minimum area in square meters. Defaults to 750.
    expected_crs_meters : str, optional
        CRS expected for area computations (meters). Defaults to "EPSG:28992".
    logger : logging.Logger | None, optional
        Optional injected logger; if None, a module/class logger is used.
    """

    def __init__(
        self,
        allowed_parking_types: Optional[Iterable[Optional[str]]] = None,
        min_parking_size: float = 750.0,
        expected_crs_meters: str = "EPSG:28992",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.min_parking_size = float(min_parking_size)
        self.expected_crs_meters = expected_crs_meters

        # Default whitelist (copied from your original, normalized)
        self.allowed_parking_types: List[Optional[str]] = (
            list(allowed_parking_types)
            if allowed_parking_types is not None
            else [
                None, "surface", "carpool", "Carpool", "service", "yes",
                "kiss_and_ride", "Parking Fitness", "Sportpark Heide", "parking",
                "caravan", "private", "residential", "separate", "Kiss+and+Ride",
                "staging",
            ]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        df: gpd.GeoDataFrame,
        folder: Union[str, Path],
        only_load: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Execute the filtering pipeline, with optional caching to a GeoJSON.

        Parameters
        ----------
        df : GeoDataFrame
            Input GeoDataFrame (projected in meters; default EPSG:28992).
        folder : str | Path
            Folder to save the processed file or load it from.
        only_load : bool, optional
            If True, loads the cached result and skips processing.

        Returns
        -------
        GeoDataFrame
            Filtered GeoDataFrame.
        """
        out_dir = Path(folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "gdf_all_processed.geojson"

        if only_load:
            if not out_file.exists():
                raise FileNotFoundError(
                    f"Requested only_load=True, but file not found: {out_file}"
                )
            self.logger.info("Loading preprocessed data from %s", out_file)
            return gpd.read_file(out_file)

        # Validate input quickly
        self._validate_input(df)

        self.logger.info("Filtering based on parking types: %s", self.allowed_parking_types)
        df_types = self.filter_parking_type(df)

        self.logger.info("Ensuring/Computing 'area_m2' and filtering with min size: %.2f m²", self.min_parking_size)
        df_sized = self.filter_parking_size(df_types)

        self.logger.info("Removing overlapping polygons (keeping the largest by area_m2)")
        df_dedup = self.filter_overlapping_geometries(df_sized)

        # Save & return
        # Ensure CRS is present; for interchange, GeoJSON typically uses WGS84 (EPSG:4326).
        # But if you prefer to keep EPSG:28992 on disk, comment the next line.
        df_out = df_dedup.to_crs(epsg=4326)
        df_out.to_file(out_file, driver="GeoJSON")
        self.logger.info("Saved processed data to %s (rows=%d)", out_file, len(df_out))

        return df_out

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    def filter_parking_type(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Keep rows where 'parking' is in allowed_parking_types (or None).
        Missing 'parking' column is treated as not allowed.
        """
        if "parking" not in df.columns:
            raise ValueError("Column 'parking' is required for type filtering.")

        before = len(df)
        mask = df["parking"].isin(self.allowed_parking_types)
        df_filtered = df.loc[mask].copy()

        self.logger.info(
            "Removed %d rows with disallowed parking types. Remaining: %d",
            before - len(df_filtered), len(df_filtered)
        )
        return df_filtered

    def filter_parking_size(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Ensure/compute 'area_m2' and keep polygons with area >= min_parking_size.

        Notes
        -----
        - Expects a projected CRS in meters (default EPSG:28992).
        - If 'area_m2' is missing, computes from geometry.
        """
        if df.empty:
            return df

        # Ensure projected CRS for area calculations
        df_proj = self._ensure_projected_for_area(df)

        # Compute area if missing
        if "area_m2" not in df_proj.columns:
            df_proj = df_proj.copy()
            df_proj["area_m2"] = df_proj.geometry.area

        # Keep rows with sufficient area (fixing the prior '&gt;' issue)
        before = len(df_proj)
        df_filtered = df_proj.loc[df_proj["area_m2"] >= self.min_parking_size].copy()

        self.logger.info(
            "Removed %d rows with area < %.2f m². Remaining: %d",
            before - len(df_filtered), self.min_parking_size, len(df_filtered)
        )
        return df_filtered

    def filter_overlapping_geometries(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Remove polygons fully covered by larger polygons (by 'area_m2'),
        keeping the largest per overlapping group.

        Strategy
        --------
        - Keep only Polygon and MultiPolygon.
        - Sort by 'area_m2' descending.
        - Use spatial index to find candidate overlaps quickly.
        - If polygon j is fully within polygon i (i larger), drop j.
        """
        if df.empty:
            return df

        # Keep only polygons
        df_poly = df[df.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        if df_poly.empty:
            self.logger.info("No polygonal geometries found; returning empty GeoDataFrame.")
            return df_poly

        # Sort by area descending (must have 'area_m2')
        if "area_m2" not in df_poly.columns:
            raise ValueError("Column 'area_m2' is required before overlap filtering.")
        df_poly = df_poly.sort_values(by="area_m2", ascending=False).reset_index(drop=True)

        # Spatial index
        try:
            sindex = df_poly.sindex
        except Exception as exc:
            self.logger.warning("Spatial index is unavailable; falling back to O(n^2). Details: %s", exc)
            sindex = None

        to_drop = set()

        for idx, geom in df_poly.geometry.items():
            if idx in to_drop or geom is None or geom.is_empty:
                continue

            # Candidate matches via spatial index or all indices if not available
            if sindex is not None:
                possible_matches = list(sindex.intersection(geom.bounds))
            else:
                possible_matches = range(len(df_poly))

            for j in possible_matches:
                if j == idx or j in to_drop:
                    continue
                other = df_poly.geometry.iloc[j]
                if other is None or other.is_empty:
                    continue

                # If smaller polygon is within the larger current polygon, drop the smaller
                if other.within(geom):
                    to_drop.add(j)

        df_filtered = df_poly.drop(index=list(to_drop)).reset_index(drop=True)
        self.logger.info("Removed %d covered polygons. Remaining: %d", len(to_drop), len(df_filtered))
        return df_filtered

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _validate_input(self, df: gpd.GeoDataFrame) -> None:
        if not isinstance(df, gpd.GeoDataFrame):
            raise TypeError("Input must be a GeoDataFrame.")
        if df.crs is None:
            self.logger.warning(
                "Input GeoDataFrame has no CRS. Assuming '%s' for area computations.",
                self.expected_crs_meters
            )

    def _ensure_projected_for_area(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Ensure the GeoDataFrame is in a metric, projected CRS (meters) for area calculations.
        Defaults to EPSG:28992 if CRS is missing or not projected.
        """
        # If CRS is missing, assign expected CRS
        if df.crs is None:
            df = df.set_crs(self.expected_crs_meters)
            return df

        # If CRS is geographic (degrees), project to expected metric CRS
        if not df.crs.is_projected:
            self.logger.info(
                "Projecting GeoDataFrame to %s for accurate area calculations.",
                self.expected_crs_meters
            )
            return df.to_crs(self.expected_crs_meters)

        # Already projected in meters
        return df
