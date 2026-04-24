"""
GenerateGemeenteResidents
-------------------------
Generate synthetic residents for Dutch gemeenten by distributing
CBS buurt populations over BAG verblijfsobjecten (woonfunctie),
using spatial joins and Python-side filtering only.
"""

from __future__ import annotations

import logging
import random
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
import requests

# =============================================================================
# Constants
# =============================================================================

BAG_CRS = "EPSG:28992"
OUTPUT_CRS = "EPSG:4326"

PDOK_BAG_VERBLIJFSOBJECTEN_URL = (
    "https://api.pdok.nl/kadaster/bag/ogc/v2/collections/verblijfsobject/items"
)


# =============================================================================
# BAG fetch helper (NO FILTERING!)
# =============================================================================

def fetch_verblijfsobjecten_bbox(
    bbox_4326: tuple[float, float, float, float],
    limit: int = 1000,
) -> gpd.GeoDataFrame:
    """
    Fetch all BAG verblijfsobjecten within a bbox (EPSG:4326).
    Filtering is done later in Python.
    """
    params: Dict[str, Any] = {
        "bbox": ",".join(map(str, bbox_4326)),
        "limit": limit,
    }

    features: List[Dict[str, Any]] = []
    next_url: Optional[str] = PDOK_BAG_VERBLIJFSOBJECTEN_URL

    while next_url:
        response = requests.get(next_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        features.extend(data.get("features", []))

        next_url = None
        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_url = link.get("href")
                params = None
                break

    gdf = gpd.GeoDataFrame.from_features(features, crs=OUTPUT_CRS)
    return gdf.to_crs(BAG_CRS)

def has_woonfunctie(value: Any) -> bool:
    """
    Return True if 'woonfunctie' is present in gebruiksdoel,
    regardless of whether the value is a string or a list.
    """
    if isinstance(value, str):
        return "woonfunctie" in value.lower()

    if isinstance(value, list):
        return any(
            isinstance(v, str) and "woonfunctie" in v.lower()
            for v in value
        )

    return False

# =============================================================================
# Main class
# =============================================================================

class GenerateGemeenteResidents:
    def __init__(
        self,
        residents_folder: Union[str, Path],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.residents_folder = Path(residents_folder)
        self.residents_folder.mkdir(parents=True, exist_ok=True)

        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        self.run_success: List[Tuple[str, bool]] = []

    # ------------------------------------------------------------------
    def run(
        self,
        gdf_gemeenten: gpd.GeoDataFrame,
        gdf_buurten: gpd.GeoDataFrame,
        overwrite: bool = False,
    ) -> str:
        if not overwrite:
            msg = "Not overwriting — nothing to do."
            self.logger.info(msg)
            return msg

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = {
                executor.submit(
                    self._process_gemeente,
                    row,
                    gdf_buurten[gdf_buurten["gemeentenaam"] == row["gemeentenaam"]],
                ): row["gemeentenaam"]
                for _, row in gdf_gemeenten.iterrows()
            }

            for future in as_completed(futures):
                gemeente = futures[future]
                try:
                    future.result()
                    self.run_success.append((gemeente, True))
                except Exception as exc:  # noqa: BLE001
                    self.logger.error("Error processing %s: %s", gemeente, exc)
                    self.run_success.append((gemeente, False))

        return "Finished synthetic resident generation."

    # ------------------------------------------------------------------
    def _process_gemeente(
        self,
        gemeente_row: pd.Series,
        buurten: gpd.GeoDataFrame,
    ) -> None:
        gemeente = str(gemeente_row["gemeentenaam"])
        out_fp = self.residents_folder / f"{gemeente.replace(' ', '_')}.geojson"

        if out_fp.exists():
            return

        self.logger.info("Processing gemeente: %s", gemeente)

        buurten = buurten.to_crs(BAG_CRS)

        # --------------------------------------------------------------
        # Fetch verblijfsobjecten via bbox
        # --------------------------------------------------------------
        buurten_4326 = buurten.to_crs(OUTPUT_CRS)
        bbox_4326 = tuple(buurten_4326.total_bounds)

        verblijfsobjecten = fetch_verblijfsobjecten_bbox(bbox_4326)
        verblijfsobjecten = verblijfsobjecten[
            verblijfsobjecten["gebruiksdoel"].apply(has_woonfunctie)
        ]

        # --------------------------------------------------------------
        # Spatial join: verblijfsobject ∈ buurt
        # --------------------------------------------------------------
        joined = gpd.sjoin(
            verblijfsobjecten,
            buurten[["buurtcode", "aantal_inwoners", "geometry"]],
            how="inner",
            predicate="within",
        )

        residents = []

        # Iterate over each neighborhood (buurtcode)
        for buurtcode, group in joined.groupby("buurtcode"):

            # Total population of the neighborhood
            # Assumed equal for all rows in this group
            pop = int(group["aantal_inwoners"].iloc[0])

            # Number of residential objects (e.g. buildings/addresses)
            n_objects = len(group)

            # Minimum number of residents per object
            base = pop // n_objects

            # Remaining residents after equal division
            remainder = pop % n_objects

            # Randomly select `remainder` object indices that get +1 resident
            extra_indices = set(random.sample(range(n_objects), remainder))

            # Iterate over each object in the neighborhood
            for idx, row in enumerate(group.itertuples()):

                # Distribute remainer: add one extra resident only for randomly selected objects
                n = base + (1 if idx in extra_indices else 0)

                for _ in range(n):
                    residents.append(
                        {
                            "geometry": row.geometry,
                            "gemeentenaam": gemeente,
                            "buurtcode": buurtcode,
                            "verblijfsobject_id": row.identificatie,
                        }
                    )

        gdf_out = gpd.GeoDataFrame(residents, crs=BAG_CRS).to_crs(OUTPUT_CRS)
        gdf_out.to_file(out_fp, driver="GeoJSON")

        self.logger.info("Saved gemeente residents: %s", out_fp)
