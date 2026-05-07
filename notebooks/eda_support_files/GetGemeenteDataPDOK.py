from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import pandas as pd
import requests


class GetGemeenteDataPDOK:
    """
    Retrieve and process CBS neighborhood/municipality data from the PDOK
    'wijken-en-buurten' OGC API.

    Workflow
    --------
    1. Optionally download features from the PDOK OGC API.
    2. Convert retrieved features into a GeoDataFrame.
    3. Select relevant columns.
    4. Remove water-surface areas and duplicate geometries.
    5. Clean codes, fill missing values, and compute population distributions.
    6. Aggregate to municipality level (gemeente) and append to the dataset.

    Parameters
    ----------
    limit : int, optional
        API paging size per request. Default is 1000 features per API call.
    filename : str | Path, optional
        File where the GeoJSON will be cached. Default is the internal constant.
    logger : logging.Logger | None, optional
        Optional injected logger. If None, uses a class-named logger.
    """

    BASE_URL = (
        "https://api.pdok.nl/cbs/wijken-en-buurten-2023/ogc/v1/collections/{collection}/items"
    )

    DEFAULT_COLUMNS = [
    "geometry", "jrstatcode", "jaar", "buurtcode", "buurtnaam",
    "gemeentecode", "gemeentenaam", "aantal_inwoners", "water"]

    def __init__(
        self,
        limit: int = 1000,
        columns: Optional[List[str]] = None,
        filepath: Union[str, Path] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:

        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.limit = int(limit)
        self.collection_type = "buurten"
        self.features: List[dict] = []
        self.filename = Path(filepath + "/gemeente.geojson")
        self.gdf: Optional[gpd.GeoDataFrame] = None

        extra_cols = columns or []
        self.columns = list(dict.fromkeys(self.DEFAULT_COLUMNS + extra_cols)) 

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, load_from_file: bool = False) -> gpd.GeoDataFrame:
        """
        Execute the full processing pipeline.

        Parameters
        ----------
        load_from_file : bool, optional
            If True, loads the cached GeoJSON instead of calling the API.

        Returns
        -------
        GeoDataFrame
            Processed PDOK data including gemeente-level aggregation.
        """
        if load_from_file:
            self._load_from_file()
        else:
            self._fetch_all()
            self._to_geodataframe()

        self._select_relevant_columns()
        self._filter_water()
        self._fill_missing_values()
        self._clean_gemeentecode()
        self._aggregate_gemeenten()

        # PDOK uses EPSG:4326; enforce it explicitly
        self.gdf = self.gdf.set_crs("EPSG:4326")

        return self.gdf

    # ------------------------------------------------------------------
    # Fetch & IO
    # ------------------------------------------------------------------
    def _load_from_file(self) -> None:
        if not self.filename.exists():
            raise FileNotFoundError(
                f"Requested load_from_file=True, but file not found: {self.filename}"
            )
        self.logger.info("Loading PDOK data from cached file: %s", self.filename)
        self.gdf = gpd.read_file(self.filename)

    def _fetch_all(self) -> None:
        """Fetch all items from the PDOK API until no 'next' link is provided."""
        url = self.BASE_URL.format(collection=self.collection_type)
        next_url = url
        params = {"limit": self.limit}

        self.logger.info("Fetching PDOK data from: %s", url)

        while next_url:
            response = requests.get(next_url, params=params if next_url == url else None)
            response.raise_for_status()

            data = response.json()
            items = data.get("features", [])
            self.features.extend(items)

            next_url = next(
                (link.get("href") for link in data.get("links", []) if link.get("rel") == "next"),
                None,
            )

        self.logger.info(
            "Fetched %d features from PDOK collection '%s'.",
            len(self.features), self.collection_type
        )

    def _to_geodataframe(self) -> None:
        """Convert features to a GeoDataFrame and save to cache."""
        if not self.features:
            raise RuntimeError("No features fetched. Call _fetch_all() first.")

        self.gdf = gpd.GeoDataFrame.from_features(self.features)
        self.gdf = self.gdf.set_crs("EPSG:4326")

        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.gdf.to_file(self.filename, driver="GeoJSON")
        self.logger.info("Saved raw PDOK data to %s", self.filename)

    # ------------------------------------------------------------------
    # Cleaning & Filtering
    # ------------------------------------------------------------------
    def _select_relevant_columns(self) -> None:
        cols = [c for c in self.DEFAULT_COLUMNS if c in self.gdf.columns]
        self.logger.info("Selecting %d relevant columns.", len(cols))
        self.gdf = self.gdf[cols]

    def _filter_water(self) -> None:
        """
        Remove features representing water areas.

        PDOK provides both land and water geometries. We keep only rows where:
          - water == 'NEE'
        """
        if "water" not in self.gdf.columns:
            self.logger.warning("Column 'water' missing; skipping water filtering.")
            return

        before = len(self.gdf)
        self.logger.info("Filtering out water-surface geometries (water == 'NEE').")

        self.gdf = self.gdf[self.gdf["water"] == "NEE"].copy()
        after = len(self.gdf)

        self.logger.info("Removed %d water geometries. Remaining: %d", before - after, after)
        self.gdf = self.gdf.drop(columns=["water"], errors="ignore")

    def _fill_missing_values(self) -> None:
        """
        Replace CBS missing-value flags:
          - -99995: No information available
          - -99997: Sample too small
        """
        replacement = {-99995: None, -99997: None}
        self.gdf = self.gdf.replace(replacement)

    def _clean_gemeentecode(self) -> None:
        """Trim whitespace in gemeentecode values."""
        if "gemeentecode" in self.gdf.columns:
            self.gdf["gemeentecode"] = self.gdf["gemeentecode"].astype(str).str.strip()

    # ------------------------------------------------------------------
    # Computations
    # ------------------------------------------------------------------

    def _aggregate_gemeenten(self) -> None:
        """
        Aggregate neighborhood-level (buurt) data into gemeente-level totals.

        Geometry is dissolved on gemeente attributes.
        """
        numeric_cols = [
            "aantal_inwoners"
        ]

        # Dissolve into municipality polygons
        gemeente_df = (
            self.gdf.dissolve(
                by=["gemeentecode", "gemeentenaam"],
                aggfunc={col: "sum" for col in numeric_cols},
            )
            .reset_index()
        )
        gemeente_df["level"] = "gemeente"

        # Mark neighborhood level
        self.gdf["level"] = "buurt"

        # Combine buurt + gemeente
        self.gdf = pd.concat([self.gdf, gemeente_df], ignore_index=True)
