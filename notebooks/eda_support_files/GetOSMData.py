import os
import logging
from pathlib import Path
from typing import List, Tuple, Union

import osmnx as ox
import geopandas as gpd
import pandas as pd

from eda_support_files.CONSTANTS import WS84_NL_BOUNDS


class GetOSMData:
    """
    Download and load OpenStreetMap (OSM) parking-related data in chunks
    based on a tile grid over the Netherlands.

    The workflow:
    1. Divide the Netherlands extent into tiles of configurable size.
    2. Download OSM features for each tile using OSMnx.
    3. Periodically write tiles to GeoJSON chunks.
    4. Load all written GeoJSON files back into memory as a single GeoDataFrame.

    Parameters
    ----------
    tile_size : float, optional
        Size of each tile in degrees for the grid. Default is 0.5.
    """

    def __init__(self, tile_size: float = 0.5) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tile_size = tile_size

        # OSM tags for parking-related POIs
        self.poi_tags = {
            "amenity": ["parking", "parking_space"]
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, folder: Union[str, Path], only_load: bool = False) -> gpd.GeoDataFrame:
        """
        Execute the OSM parking data pipeline.

        Parameters
        ----------
        folder : str | Path
            Directory where GeoJSON chunks are stored and/or loaded from.
        only_load : bool, optional
            If True, skip downloading and load existing files only.

        Returns
        -------
        GeoDataFrame
            Combined parking-related data from OSM.
        """
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)

        if not only_load:
            self._download_in_chunks(output_dir=folder_path)

        files = [f for f in folder_path.iterdir() if f.name.startswith("netherlands_parking_part")]
        return self._load_data(files)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_in_chunks(self, output_dir: Path) -> None:
        """Download OSM data tile-by-tile and write GeoJSON chunks."""

        tiles = self._divide_in_tiles()
        buffer: List[gpd.GeoDataFrame] = []

        for idx, (north, south, east, west) in enumerate(tiles):
            tile_number = idx + 1

            try:
                gdf_tile = ox.features_from_bbox(
                    bbox=(west, south, east, north),
                    tags=self.poi_tags
                )
                buffer.append(gdf_tile)
                self.logger.info("Downloaded tile %d/%d", tile_number, len(tiles))

            except Exception as exc:
                # Common for water areas (e.g., Noordzee, IJsselmeer)
                self.logger.warning("Tile %d returned no data: %s", tile_number, exc)

            # Save every 12 tiles OR when last tile is reached
            should_save = (tile_number % 12 == 0) or (tile_number == len(tiles))
            if should_save:
                chunk_index = (tile_number // 12) if (tile_number % 12 == 0) else (tile_number // 12) + 1

                gdf_chunk = pd.concat(buffer).reset_index(drop=True)
                gdf_chunk = gdf_chunk.to_crs(epsg=4326)

                # Select meaningful columns (drop all-NaN columns implicitly)
                columns = ["geometry", "access", "parking", "amenity", "capacity", "name", "surface", "operator"]
                available_columns = [col for col in columns if col in gdf_chunk.columns]
                gdf_chunk = gdf_chunk[available_columns]

                filename = output_dir / f"netherlands_parking_part{chunk_index}.geojson"
                gdf_chunk.to_file(filename, driver="GeoJSON")

                self.logger.info(
                    "Saved chunk %d containing %d tiles → %s",
                    chunk_index, len(buffer), filename
                )

                buffer = []  # Reset for next group

    def _load_data(self, files: List[Path]) -> gpd.GeoDataFrame:
        """Load GeoJSON chunk files into memory and merge into a single GeoDataFrame."""
        if not files:
            raise FileNotFoundError("No chunk files found to load.")

        gdfs = [gpd.read_file(file) for file in files]
        gdf_all = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True),
            crs=gdfs[0].crs if gdfs else None
        ).to_crs(epsg=28992)

        self.logger.info("Loaded %d files into a single GeoDataFrame.", len(files))
        return gdf_all

    def _divide_in_tiles(self) -> List[Tuple[float, float, float, float]]:
        """
        Divide the bounding box of the Netherlands into tiles of size `tile_size`.

        Returns
        -------
        list of (north, south, east, west)
        """

        tiles = []
        lat_steps = int((WS84_NL_BOUNDS["north"] - WS84_NL_BOUNDS["south"]) / self.tile_size) + 1
        lon_steps = int((WS84_NL_BOUNDS["east"] - WS84_NL_BOUNDS["west"]) / self.tile_size) + 1

        for i in range(lat_steps):
            for j in range(lon_steps):
                north = WS84_NL_BOUNDS["south"] + (i + 1) * self.tile_size
                south = WS84_NL_BOUNDS["south"] + i * self.tile_size
                east = WS84_NL_BOUNDS["west"] + (j + 1) * self.tile_size
                west = WS84_NL_BOUNDS["west"] + j * self.tile_size
                tiles.append((north, south, east, west))

        return tiles
