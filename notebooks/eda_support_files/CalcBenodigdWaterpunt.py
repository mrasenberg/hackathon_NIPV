import logging
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd


class CalcBenodigdWaterpunt:
    """
    Calculate the required number of emergency water points based on the number
    of inhabitants in each area.

    For each row:
        Benodigd_ceiling = ceil(aantal_inwoners / MAX_CITIZEN_PER_POINT)

    Parameters
    ----------
    max_citizens_per_point : int, optional
        Maximum number of inhabitants served by a single water point.
        Default is 2500.
    """

    def __init__(self, max_citizens_per_point: int = 2500) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_citizens_per_point = int(max_citizens_per_point)

    def run(self, gdf: Union[gpd.GeoDataFrame, pd.DataFrame]) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
        """
        Add a column 'Benodigd_ceiling' indicating required emergency water points.

        Parameters
        ----------
        gdf : GeoDataFrame or DataFrame
            Input dataset containing column 'aantal_inwoners'.

        Returns
        -------
        GeoDataFrame or DataFrame
            The input DataFrame with new column 'Benodigd_ceiling'.

        Raises
        ------
        ValueError
            If 'aantal_inwoners' column is missing.
        """

        if "aantal_inwoners" not in gdf.columns:
            raise ValueError(
                "Column 'aantal_inwoners' is required for calculating water points."
            )

        self.logger.info(
            "Calculating required water points (max %d citizens per point).",
            self.max_citizens_per_point
        )

        gdf = gdf.copy()
        gdf["Benodigd_ceiling"] = np.ceil(
            gdf["aantal_inwoners"] / self.max_citizens_per_point
        ).astype("Int64")  # nullable integer type

        self.logger.info(
            "Added 'Benodigd_ceiling' column for %d rows.", len(gdf)
        )

        return gdf
