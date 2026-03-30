from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, Tuple, List

import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Empty

from .DistanceBasedLotSelector import (
    EvenSpreadSelector,
    MinAvgDistanceSelector,
    MinMaxDistanceSelector,
)
from .RandomLotSelector import RandomSelector
from .ResidentBasedLotSelector import (
    ClusteringSelector,
    ResidentDensitySelector,
)
from .assign_residents import assign_residents_to_parking_lots


# --------------------------------------------------------------------------------------
# Strategy registry (unchanged)
# --------------------------------------------------------------------------------------
STRATEGY_REGISTRY = {
    "RandomSelector": RandomSelector,
    "EvenSpreadSelector": EvenSpreadSelector,
    "ResidentDensitySelector": ResidentDensitySelector,
    "MinAvgDistanceSelector": MinAvgDistanceSelector,
    "MinMaxDistanceSelector": MinMaxDistanceSelector,
    "ClusteringSelector": ClusteringSelector,
}


# --------------------------------------------------------------------------------------
# Event dataclass
# --------------------------------------------------------------------------------------
@dataclass
class Event:
    kind: str               # "start", "update", "end", "error"
    gemeente: str
    message: Optional[str] = None
    payload: Optional[Any] = None


# --------------------------------------------------------------------------------------
# Main orchestrator
# --------------------------------------------------------------------------------------
class ExperimentOrchestrator:
    """
    Orchestrates experiments for multiple municipalities, running both
    municipalities and experiments in parallel.

    Features
    --------
    - Clean logging hierarchy
    - Configurable workers for municipalities & experiments
    - Structured event handling
    - Per-gemeente logs stored in: output_folder/logs/gemeente/<Gemeente>.txt
    - Clear validation and flow structure
    """

    def __init__(
        self,
        gemeente_filepaths: Iterable[str],
        setup_experiments: Dict[str, Dict[str, Any]],
        output_folder: str,
        gdf_parking_lots: gpd.GeoDataFrame,
        gdf_gemeenten: gpd.GeoDataFrame,
        projected_crs: str = "EPSG:28992",
        max_workers_gemeenten: int = 3,
        max_workers_experiments: int = 4,
        enable_logging: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:

        self.gemeente_filepaths = list(gemeente_filepaths)
        self.setup_experiments = setup_experiments

        # Directories
        self.output_folder = Path(output_folder)
        self.logs_dir = self.output_folder / "logs" / "gemeente"

        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # CRS
        self.PROJECTED_CRS = projected_crs
        self.gdf_parking_lots = gdf_parking_lots.to_crs(self.PROJECTED_CRS)
        self.gdf_gemeenten = gdf_gemeenten.to_crs(self.PROJECTED_CRS)

        # Threading
        self.max_workers_gemeenten = max_workers_gemeenten
        self.max_workers_experiments = max_workers_experiments

        # Logging
        self.enable_logging = enable_logging
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    # ----------------------------------------------------------------------------------
    # Utils
    # ----------------------------------------------------------------------------------
    def _gemeente_name_from_fp(self, fp: str) -> str:
        """Extract gemeente name from filename."""
        return os.path.basename(fp).replace(".geojson", "").replace("_", " ")

    def _get_gemeente_logger(self, gemeente: str) -> logging.Logger:
        """
        Create a clean logger for a single gemeente.
        Logs to output_folder/logs/gemeente/<Gemeente>.txt.
        """
        logger = logging.getLogger(f"orchestrator.{gemeente}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        for h in list(logger.handlers):
            logger.removeHandler(h)

        if self.enable_logging:
            logfile = self.logs_dir / f"{gemeente}.txt"
            fh = logging.FileHandler(logfile)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(fh)

        return logger

    # ==================================================================================
    # Per-gemeente worker
    # ==================================================================================
    def _worker_run_gemeente(
        self,
        gemeente_name: str,
        gdf_residents: gpd.GeoDataFrame,
        event_queue: Queue,
    ) -> str:
        """
        Run all experiments for a single municipality.
        """
        gemeente_logger = self._get_gemeente_logger(gemeente_name)

        results_dir = self.output_folder / gemeente_name
        results_dir.mkdir(parents=True, exist_ok=True)

        event_queue.put(Event("start", gemeente_name, f"{len(self.setup_experiments)} experiments"))
        gemeente_logger.info("=== Start gemeente: %s ===", gemeente_name)

        # Extract only matching records
        gdf_pl = self.gdf_parking_lots[self.gdf_parking_lots["gemeentenaam"] == gemeente_name]
        gdf_gm = self.gdf_gemeenten[self.gdf_gemeenten["gemeentenaam"] == gemeente_name]

        if gdf_pl.empty or gdf_gm.empty:
            msg = "No parking or municipality data available."
            gemeente_logger.warning(msg)
            event_queue.put(Event("end", gemeente_name, msg))
            return gemeente_name

        # ------------------------------------------------------------------
        # Experiment worker
        # ------------------------------------------------------------------
        def _run_experiment(exp_name: str, setup: dict):
            output_file = results_dir / f"{exp_name}.geojson"

            if output_file.exists():
                msg = f"Skipping existing result: {exp_name}"
                gemeente_logger.info(msg)
                event_queue.put(Event("update", gemeente_name, msg))
                return

            strategy_key = setup.get("optimisation_class")
            strategy_class = STRATEGY_REGISTRY.get(strategy_key)

            if strategy_class is None:
                errmsg = f"Unknown strategy: {strategy_key}"
                gemeente_logger.error(errmsg)
                event_queue.put(Event("update", gemeente_name, errmsg))
                raise ValueError(errmsg)

            selector = strategy_class(gdf_gm, gdf_residents, gdf_pl)
            method = setup.get("assignment_method", "closest")

            event_queue.put(Event("update", gemeente_name, f"Running {exp_name}"))
            gemeente_logger.info("Running experiment: %s (%s)", exp_name, method)

            try:
                if method != "in_optimisation":
                    selected = selector.select_parking_lots()
                    assign_residents_to_parking_lots(
                        gdf_residents,
                        selected,
                        output_file=output_file,
                        mode=method,
                        max_capacity=2500,
                    )
                else:
                    selector.select_parking_lots(output_file=output_file)

            except Exception as exc:
                errmsg = f"{exp_name} failed: {exc}"
                gemeente_logger.error(errmsg)
                event_queue.put(Event("error", gemeente_name, errmsg))
                raise

        # ------------------------------------------------------------------
        # Run experiments in parallel
        # ------------------------------------------------------------------
        with ThreadPoolExecutor(max_workers=self.max_workers_experiments) as executor:
            future_to_name = {
                executor.submit(_run_experiment, name, setup): name
                for name, setup in self.setup_experiments.items()
            }

            for fut, name in future_to_name.items():
                try:
                    fut.result()
                except Exception:
                    # Already logged inside _run_experiment
                    pass

        gemeente_logger.info("=== Done gemeente: %s ===", gemeente_name)
        event_queue.put(Event("end", gemeente_name, "Completed"))
        return gemeente_name

    # ==================================================================================
    # MAIN RUN LOOP
    # ==================================================================================
    def run(self) -> None:
        """
        Run the full multi-gemeente orchestration pipeline.
        """
        if not self.gemeente_filepaths:
            self.logger.error("No municipality files provided.")
            return

        # ------------------------------------------------------------------
        # Load residents
        # ------------------------------------------------------------------
        jobs: List[Tuple[str, gpd.GeoDataFrame]] = []
        for fp in self.gemeente_filepaths:
            name = self._gemeente_name_from_fp(fp)
            try:
                gdf_res = gpd.read_file(fp).to_crs(self.PROJECTED_CRS)
                jobs.append((name, gdf_res))
            except Exception as exc:
                self.logger.error("Failed to read %s: %s", name, exc)

        if not jobs:
            self.logger.error("No resident data loaded — aborting.")
            return

        # ------------------------------------------------------------------
        # Parallel run over gemeenten
        # ------------------------------------------------------------------
        self.logger.info("Starting orchestration with %d gemeente workers.", self.max_workers_gemeenten)
        event_queue: Queue = Queue()

        with ThreadPoolExecutor(max_workers=self.max_workers_gemeenten) as executor:
            future_to_name: Dict[Future, str] = {
                executor.submit(self._worker_run_gemeente, name, gdf_res, event_queue): name
                for name, gdf_res in jobs
            }

            completed = 0
            total = len(future_to_name)

            self.logger.info("Processing %d gemeenten...", total)

            # Event loop
            while completed < total:
                # Process queued events
                try:
                    while True:
                        ev: Event = event_queue.get_nowait()

                        if ev.kind == "start":
                            print(f"[{ev.gemeente}] Starting ({ev.message})")
                        elif ev.kind == "update":
                            print(f"[{ev.gemeente}] → {ev.message}")
                        elif ev.kind == "error":
                            print(f"[ERROR: {ev.gemeente}] {ev.message}")
                        elif ev.kind == "end":
                            print(f"[{ev.gemeente}] Finished: {ev.message}")

                except Empty:
                    pass

                # Check completed futures
                for fut in list(future_to_name.keys()):
                    if fut.done():
                        gemeente = future_to_name.pop(fut)
                        try:
                            fut.result()
                        except Exception as exc:
                            print(f"[ERROR] {gemeente}: {exc}")
                        completed += 1

                time.sleep(0.05)

        print("\nAll municipalities processed.")