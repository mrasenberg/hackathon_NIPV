import numpy as np
import geopandas as gpd


class ParkingLotSelector:
    def __init__(self, gdf_gemeente, gdf_residents, gdf_parking_lots):
        # Always work on copies to avoid chained assignment issues
        self.gdf_gemeente = gdf_gemeente.copy()
        self.gdf_residents = gdf_residents.copy()
        self.gdf_parking_lots = gdf_parking_lots.copy()

        # Safely extract required_count
        self.required_count = int(self.gdf_gemeente.loc[:, "Benodigd_ceiling"].iloc[0])

        # Compute centroids and store them explicitly using .loc
        self.gdf_parking_lots.loc[:, "centroid"] = self.gdf_parking_lots.geometry.centroid

        # Precompute lot coordinates
        self.lot_coords = np.array(
            [[pt.x, pt.y] for pt in self.gdf_parking_lots.loc[:, "centroid"]]
        )

    def compute_distance_matrix(self, residents, lots):
        # Ensure copies to avoid modifying views
        residents = residents.copy()
        lots = lots.copy()

        res_coords = np.array([[pt.x, pt.y] for pt in residents.geometry])
        lot_coords = np.array([[pt.x, pt.y] for pt in lots.geometry])

        return np.linalg.norm(res_coords[:, None] - lot_coords[None, :], axis=2)

import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
from typing import Optional


class ParkingLotSelector:
    """
    Utility class for selecting and analyzing parking lots based on resident locations.

    This class:
    - Stores copies of the input GeoDataFrames to avoid side effects.
    - Extracts a required parking capacity from the municipality GeoDataFrame.
    - Precomputes parking lot centroids and their coordinates.
    - Provides a method to compute distance matrices between residents and lots.

    All geometries are assumed to be in a projected CRS with metric units
    (e.g. EPSG:28992), so Euclidean distances are interpretable as meters.

    Parameters
    ----------
    gdf_gemeente : geopandas.GeoDataFrame
        GeoDataFrame containing municipality-level information.
        Must contain a column ``"Benodigd_ceiling"`` where the first row
        represents the required capacity.
    gdf_residents : geopandas.GeoDataFrame
        GeoDataFrame containing resident locations. The active geometry column
        must contain point geometries in the same projected CRS.
    gdf_parking_lots : geopandas.GeoDataFrame
        GeoDataFrame containing parking lot geometries (points or polygons)
        in the same projected CRS. Centroids are used for distance calculations.

    Attributes
    ----------
    gdf_gemeente : geopandas.GeoDataFrame
        Copy of the input municipality GeoDataFrame.
    gdf_residents : geopandas.GeoDataFrame
        Copy of the input residents GeoDataFrame.
    gdf_parking_lots : geopandas.GeoDataFrame
        Copy of the input parking lots GeoDataFrame with an additional
        ``"centroid"`` column containing point geometries.
    required_count : int
        Required capacity extracted from
        ``gdf_gemeente["Benodigd_ceiling"].iloc[0]``.
    lot_coords : numpy.ndarray
        Array of shape (n_lots, 2) containing (x, y) coordinates of each
        parking lot centroid, in the order of ``gdf_parking_lots``.
    """

    def __init__(
        self,
        gdf_gemeente: GeoDataFrame,
        gdf_residents: GeoDataFrame,
        gdf_parking_lots: GeoDataFrame,
    ) -> None:
        # Work on copies to avoid side effects and chained assignment issues
        self.gdf_gemeente: GeoDataFrame = gdf_gemeente.copy()
        self.gdf_residents: GeoDataFrame = gdf_residents.copy()
        self.gdf_parking_lots: GeoDataFrame = gdf_parking_lots.copy()

        # Extract the required capacity from the first row
        self.required_count: int = int(
            self.gdf_gemeente.loc[:, "Benodigd_ceiling"].iloc[0]
        )

        # Compute centroids and store them in a dedicated column
        # For polygons/multipolygons, this yields a point in the same CRS
        self.gdf_parking_lots.loc[:, "centroid"] = (
            self.gdf_parking_lots.geometry.centroid
        )

        # Precompute centroid coordinates for fast numeric operations
        self.lot_coords: np.ndarray = np.array(
            [[pt.x, pt.y] for pt in self.gdf_parking_lots.loc[:, "centroid"]]
        )

    def compute_distance_matrix(
        self,
        residents: GeoDataFrame,
        lots: GeoDataFrame,
    ) -> np.ndarray:
        """
        Compute a pairwise Euclidean distance ("diagonal distance") matrix between residents and lots.

        Distances are based on the active geometry column of the provided
        GeoDataFrames. Both are assumed to be in a metric projected CRS
        (e.g. EPSG:28992), so the resulting distances are in those same units.

        Parameters
        ----------
        residents : geopandas.GeoDataFrame
            GeoDataFrame with point geometries representing resident locations.
        lots : geopandas.GeoDataFrame
            GeoDataFrame with point geometries representing parking lot
            locations (e.g. centroids).

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (n_residents, n_lots), where entry (i, j) is the
            Euclidean distance between resident i and parking lot j.
        """
        # Work on copies to avoid mutating the caller's GeoDataFrames
        residents = residents.copy()
        lots = lots.copy()

        # Extract coordinates as (n, 2) arrays
        res_coords: np.ndarray = np.array(
            [[pt.x, pt.y] for pt in residents.geometry]
        )
        lot_coords: np.ndarray = np.array(
            [[pt.x, pt.y] for pt in lots.geometry]
        )

        # Broadcast to compute pairwise Euclidean distances
        # res_coords: (n_residents, 2)
        # lot_coords: (n_lots, 2)
        # result:     (n_residents, n_lots)
        distances: np.ndarray = np.linalg.norm(
            res_coords[:, None] - lot_coords[None, :],
            axis=2,
        )

        return distances
