"""
GenerateGemeenteResidents
-------------------------

This module generates synthetic resident-level point data for each gemeente by:

  - Sampling residents spatially using **Poisson Disk Sampling** inside each buurt
  - Assigning demographic attributes probabilistically based on buurt-level statistics
  - Saving each gemeente's synthetic residents as a GeoJSON file

Poisson Disk Sampling is based on:
    Bridson, Robert. 
    "Fast Poisson Disk Sampling in Arbitrary Dimensions."
    SIGGRAPH 2007 Sketches. ACM, 2007.
    doi:10.1145/1278780.1278807

This implementation uses a geometry-aware adaptation of Bridson's algorithm for polygons.
"""

from __future__ import annotations

import logging
import multiprocessing
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from IPython.display import display, HTML
from shapely.geometry import Point
from shapely.prepared import prep


# =============================================================================
# Helper conversion utilities
# =============================================================================

def safe_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def make_probabilities(values: List[float], population: int) -> List[float]:
    """
    Convert category counts to normalized probabilities that sum to 1.

    Steps:
      1. Replace Nones with 0
      2. Convert absolute counts to proportions
      3. Normalize if sum > 1
      4. Round
      5. Fix rounding drift so probabilities sum to 1
    """
    values = [0 if v is None else v for v in values]

    probs = [v / population for v in values]
    s = sum(probs)

    if s > 1:
        probs = [p / s for p in probs]

    probs = [round(p, 5) for p in probs]

    drift = round(1 - sum(probs), 5)
    max_idx = probs.index(max(probs))
    probs[max_idx] = round(probs[max_idx] + drift, 5)

    return probs


# =============================================================================
# Poisson Disk Sampling (Bridson 2007)
# =============================================================================

def poisson_sample_in_polygon(polygon, n_points: int, k: int = 30) -> List[Point]:
    """
    Geometry-aware Poisson Disk Sampling inside an arbitrary polygon.
    Points are distributed so that no two dots are too close together.
    Poisson Disk Sampling: efficient, but with minimal distance between points.


    Based on:
        Bridson, Robert. "Fast Poisson Disk Sampling in Arbitrary Dimensions."
        SIGGRAPH 2007 Sketches.
        https://dl.acm.org/doi/10.1145/1278780.1278807

    Parameters
    ----------
    polygon : shapely Polygon or MultiPolygon
        Area in which to sample points
    n_points : int
        Approximate number of samples (final count may differ slightly)
    k : int
        Number of attempts per active point

    Returns
    -------
    list[Point]
        Sampled points
    """
    area = polygon.area
    if area <= 0:
        return []

    # Expected area per resident
    area_per_point = area / n_points
    r = math.sqrt(area_per_point) * 0.7  # Good density tuning factor

    # Prepared geometry = geometry with added information such as an index on the line segments
    # This improves the performance of the following operations: contains, contains_properly, covered_by, covers, crosses, disjoint, intersects, overlaps, touches, and within.
    prepared = prep(polygon)

    # Grid cell resolution
    cell_size = r / math.sqrt(2)
    minx, miny, maxx, maxy = polygon.bounds

    cols = int((maxx - minx) / cell_size) + 1
    rows = int((maxy - miny) / cell_size) + 1

    grid = [[None for _ in range(cols)] for _ in range(rows)]

    def grid_coords(pt):
        gx = int((pt.x - minx) / cell_size)
        gy = int((pt.y - miny) / cell_size)
        return gy, gx

    # ----------------------------------------------------------------------
    # 1. Initial point
    # ----------------------------------------------------------------------
    while True:
        p0 = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if prepared.contains(p0):
            break

    samples = [p0]
    active = [p0]

    gy, gx = grid_coords(p0)
    if 0 <= gy < rows and 0 <= gx < cols:
        grid[gy][gx] = p0

    # ----------------------------------------------------------------------
    # 2. Bridson sampling loop
    # ----------------------------------------------------------------------
    while active and len(samples) < n_points:
        idx = random.randrange(len(active))
        center = active[idx]

        found = False
        for _ in range(k):
            rad = random.uniform(r, 2 * r)
            theta = random.uniform(0, 2 * math.pi)
            px = center.x + rad * math.cos(theta)
            py = center.y + rad * math.sin(theta)
            candidate = Point(px, py)

            if not prepared.contains(candidate):
                continue

            cy, cx = grid_coords(candidate)

            # Neighbor search
            ok = True
            for dy in [-2, -1, 0, 1, 2]:
                for dx in [-2, -1, 0, 1, 2]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < rows and 0 <= nx < cols:
                        neighbor = grid[ny][nx]
                        if neighbor is not None and neighbor.distance(candidate) < r:
                            ok = False
                            break
                if not ok:
                    break

            if ok:
                samples.append(candidate)
                active.append(candidate)
                if 0 <= cy < rows and 0 <= cx < cols:
                    grid[cy][cx] = candidate
                found = True
                break

        if not found:
            active.pop(idx)

    return samples[:n_points]


# =============================================================================
# Generation class
# =============================================================================

class GenerateGemeenteResidents:
    """
    Generate synthetic resident-level points for each gemeente using
    Poisson Disk Sampling + probabilistic attribute assignment.

    Parameters
    ----------
    residents_folder : str | Path
        Location where gemeente GeoJSON files will be saved.
    logger : logging.Logger | None
        Optional injection; defaults to class-level logger.
    """

    def __init__(self, residents_folder: Union[str, Path], logger: Optional[logging.Logger] = None) -> None:
        self.residents_folder = Path(residents_folder)
        self.residents_folder.mkdir(parents=True, exist_ok=True)

        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        self.run_success: List[Tuple[str, bool]] = []

    # ----------------------------------------------------------------------
    # Main public method
    # ----------------------------------------------------------------------
    def run(self, gdf_gemeenten: gpd.GeoDataFrame, gdf_buurten: gpd.GeoDataFrame, overwrite: bool = False):
        """
        Generate residents for all gemeenten.

        Parameters
        ----------
        gdf_gemeenten : GeoDataFrame
        gdf_buurten : GeoDataFrame
        overwrite : bool
            If True, re-generates all gemeente files.
        """
        if not overwrite:
            msg = "Not overwriting — nothing to do."
            self.logger.info(msg)
            return msg

        max_workers = multiprocessing.cpu_count()
        self.logger.info("Using %d threads for gemeente-level processing...", max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_gemeente,
                    row,
                    gdf_buurten[gdf_buurten["gemeentenaam"] == row["gemeentenaam"]],
                ): row["gemeentenaam"]
                for _, row in gdf_gemeenten.iterrows()
            }

            for fut in as_completed(futures):
                gemeente = futures[fut]
                try:
                    fut.result()
                    self.run_success.append((gemeente, True))
                except Exception as exc:
                    self.logger.error("Error processing %s: %s", gemeente, exc)
                    self.run_success.append((gemeente, False))

        return "Finished synthetic resident generation."

    # ----------------------------------------------------------------------
    # Per-gemeente processing
    # ----------------------------------------------------------------------
    def _process_gemeente(
        self,
        gemeente_row: pd.Series,
        buurten_in_gemeente: gpd.GeoDataFrame,
    ):
        gemeente_name = str(gemeente_row["gemeentenaam"])
        out_fp = self.residents_folder / f"{gemeente_name.replace(' ', '_')}.geojson"

        if out_fp.exists():
            self.logger.info("Skipping already existing gemeente: %s", gemeente_name)
            return

        self.logger.info("Processing gemeente: %s", gemeente_name)
        display(HTML(f"<br>Running {gemeente_name}..."))

        buurten_output = []

        for _, buurt in buurten_in_gemeente.iterrows():
            pop = safe_int(buurt["aantal_inwoners"])
            if pop <= 0:
                continue

            buurten_output.append(self._generate_residents_for_buurt(gemeente_name, buurt, pop))

        if not buurten_output:
            raise ValueError(f"No valid buurten found for gemeente {gemeente_name}")

        gdf = pd.concat(buurten_output).to_crs(epsg=4326)
        gdf.to_file(out_fp, driver="GeoJSON")
        self.logger.info("Saved gemeente residents: %s", out_fp)

    # ----------------------------------------------------------------------
    # Per-buurt generation
    # ----------------------------------------------------------------------
    def _generate_residents_for_buurt(self, gemeente_name: str, buurt: pd.Series, population: int) -> gpd.GeoDataFrame:
        geom = buurt.geometry

        # === Poisson Sampling ===
        points = poisson_sample_in_polygon(polygon = geom, n_points = population)

        df = gpd.GeoDataFrame(
            [{"geometry": p, "gemeentenaam": gemeente_name} for p in points],
            crs="EPSG:4326",
        )

        # # === Randomly assign demographics to each point (residents) ===
        # df["migratieachtergrond"] = self._assign_migratieachtergrond(buurt = buurt, population = population, size = len(df))
        # df["leeftijd"] = self._assign_leeftijd(buurt = buurt, population = population, size = len(df))
        # df["onderwijs_niveau"] = self._assign_onderwijs(buurt = buurt, population = population, size = len(df))
        # df["arbeids_relatie"] = self._assign_arbeid(buurt = buurt, population = population, size = len(df))
        # df["inkomens_niveau"] = self._assign_inkomen(buurt = buurt, population = population, size =len(df))
        # df["uitkering"] = self._assign_uitkering(buurt = buurt, population = population, size = len(df))
        # df["wmo_client"] = self._assign_wmo(buurt = buurt, population = population, size = len(df))

        return df

    # ----------------------------------------------------------------------
    # Assigner functions
    # ----------------------------------------------------------------------
    def _assign_migratieachtergrond(self, buurt, population: int, size: int):
        native = safe_int(buurt["Bevolking naar herkomst - Herkomstland - Nederland"])
        migr = safe_int(buurt["Bevolking naar herkomst - Herkomstland - Europa (exclusief Nederland)"]) \
             + safe_int(buurt["Bevolking naar herkomst - Herkomstland - Buiten Europa"])

        p_native, p_migr = make_probabilities(values = [native, migr], population = population)
        return np.random.choice([False, True], size=size, p=[p_native, p_migr])

    def _assign_leeftijd(self, buurt, population: int, size: int):
        oud = safe_int(buurt["personen_65_jaar_en_ouder"])
        jong = safe_int(buurt["personen_0_tot_15_jaar"])
        overige = max(0, population - oud - jong)

        p_oud, p_jong, p_overige = make_probabilities(values = [oud, jong, overige], population = population)
        return np.random.choice(
            ["vanaf_65", "tot_15", "overige"],
            size=size,
            p=[p_oud, p_jong, p_overige],
        )

    def _assign_onderwijs(self, buurt, population: int, size: int):
        o1 = safe_int(buurt["Onderwijs - Hoogst behaald onderwijsniveau - Basisonderwijs, vmbo, mbo1"])
        o2 = safe_int(buurt["Onderwijs - Hoogst behaald onderwijsniveau - Havo, vwo, mbo2-4"])
        o3 = safe_int(buurt["Onderwijs - Hoogst behaald onderwijsniveau - Hbo, wo"])
        unknown = max(0, population - (o1 + o2 + o3))

        p1, p2, p3, p_unknown = make_probabilities(values = [o1, o2, o3, unknown], population = population)
        return np.random.choice(
            ["(v)(m)bo(1)", "(ha)v(w)o|mbo2-4", "(hb)(w)o", "onbekend"],
            size=size,
            p=[p1, p2, p3, p_unknown],
        )

    def _assign_arbeid(self, buurt, population: int, size: int):
        vast = safe_float(buurt["Arbeid - Onderverdeling werkenden - Werknemers met vaste arbeidsrelatie"]) / 100 * population
        flex = safe_float(buurt["Arbeid - Onderverdeling werkenden - Werknemers met flexibele arbeidsrelatie"]) / 100 * population
        rest = max(0, population - vast - flex)

        p_vast, p_flex, p_rest = make_probabilities(values = [vast, flex, rest], population = population)
        return np.random.choice(
            ["vast_contract", "flexibel_contract", "zelfstandig"],
            size=size,
            p=[p_vast, p_flex, p_rest],
        )

    def _assign_inkomen(self, buurt, population: int, size: int):
        hoog = safe_float(buurt["Inkomen - Inkomen van personen - 20% personen met hoogste inkomen"]) / 100 * population
        laag = safe_float(buurt["Inkomen - Inkomen van personen - 40% personen met laagste inkomen"]) / 100 * population
        rest = max(0, population - hoog - laag)

        p_hoog, p_laag, p_rest = make_probabilities(values = [hoog, laag, rest], population = population)
        return np.random.choice(
            ["hoogste_20", "laagste_40", "rest"],
            size=size,
            p=[p_hoog, p_laag, p_rest],
        )

    def _assign_uitkering(self, buurt, population: int, size: int):
        ww = safe_int(buurt["Sociale zekerheid - Personen per soort uitkering; WW"])
        ao = safe_int(buurt["Sociale zekerheid - Personen per soort uitkering; AO"])
        bij = safe_int(buurt["Sociale zekerheid - Personen per soort uitkering; Bijstand"])
        none = max(0, population - (ww + ao + bij))

        p_ww, p_ao, p_bij, p_none = make_probabilities(values = [ww, ao, bij, none], population = population)
        return np.random.choice(
            ["ww", "ao", "bijstand", "niets"],
            size=size,
            p=[p_ww, p_ao, p_bij, p_none],
        )

    def _assign_wmo(self, buurt, population: int, size: int):
        yes = safe_int(buurt["Zorg - Wmo-cliënten"])
        no = max(0, population - yes)

        p_yes, p_no = make_probabilities(values = [yes, no], population = population)
        return np.random.choice([True, False], size=size, p=[p_yes, p_no])
