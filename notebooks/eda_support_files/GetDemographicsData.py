from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

try:
    import geopandas as gpd
    GeoDataFrame = gpd.GeoDataFrame
except Exception:
    gpd = None
    GeoDataFrame = pd.DataFrame  # Fallback for type hints


class DemographicsBuilder:
    """
    Build a demographics-enriched dataset by loading StatLine observations,
    pivoting measures to wide format, constructing hierarchical labels
    (Parent - Group - Measure), and merging them onto an input GeoDataFrame.

    The workflow follows:
      1. Load Observations.csv
      2. Normalize decimal separators
      3. Pivot to wide format
      4. Load MeasureCodes & MeasureGroups
      5. Construct hierarchical labels
      6. Merge onto left_gdf (GeoDataFrame)
      7. Select required output columns (fail-fast)
      8. Write resulting CSV

    Parameters
    ----------
    statline_dir : str | Path
        Directory containing the StatLine CSV files.
    output_csv : str | Path
        Output path for the final dataset.
    logger : logging.Logger | None
        Optional custom logger. If None, __name__ logger is used.
    """

    def __init__(
        self,
        statline_dir: Union[str, Path],
        output_csv: Union[str, Path],
        logger: Optional[logging.Logger] = None
    ) -> None:

        self.statline_dir = Path(statline_dir)
        self.output_csv = Path(output_csv)
        self.logger = logger or logging.getLogger(__name__)

    def run(self, left_gdf: GeoDataFrame) -> GeoDataFrame:
        """
        Execute the full demographic enrichment pipeline.

        Parameters
        ----------
        left_gdf : GeoDataFrame
            Input GeoDataFrame with buurtinformatie.

        Returns
        -------
        GeoDataFrame
            The merged demographics dataset.
        """

        self._validate_input_gdf(left_gdf)
        obs = self._load_observations()
        wide = self._pivot_observations(obs)
        label_map = self._build_labels()
        wide = wide.rename(columns=label_map)

        merged = self._merge(left_gdf, wide)
        merged = self._select_output_columns(merged)

        merged.to_csv(self.output_csv, index=False)
        self.logger.info("Wrote merged dataset to: %s", self.output_csv)

        return merged

    # ------------------------------------------------------------
    #                     INTERNAL HELPERS
    # ------------------------------------------------------------

    def _validate_input_gdf(self, gdf: GeoDataFrame) -> None:
        """Fail-fast validation of required columns in input GeoDataFrame."""
        required = [
            "geometry", "buurtnaam", "gemeentenaam",
            "aantal_inwoners", "personen_0_tot_15_jaar",
            "personen_15_tot_25_jaar", "personen_65_jaar_en_ouder",
            "aantal_huishoudens", "personenautos_totaal",
            "level", "buurtcode"
        ]

        missing = [c for c in required if c not in gdf.columns]
        if missing:
            raise ValueError(
                f"Input GeoDataFrame is missing required columns: {missing}"
            )

    # ---- Load Observations ---------------------------------------------------

    def _load_observations(self) -> pd.DataFrame:
        """Load and clean Observations.csv."""
        path = self.statline_dir / "Observations.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        obs = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)

        expected = {"WijkenEnBuurten", "Measure", "Value"}
        if not expected.issubset(obs.columns):
            raise ValueError(
                f"Observations must contain {expected}. Found: {list(obs.columns)}"
            )

        obs = obs[["WijkenEnBuurten", "Measure", "Value"]].copy()
        obs["WijkenEnBuurten"] = obs["WijkenEnBuurten"].str.strip()
        obs["Measure"] = obs["Measure"].str.strip()

        # Normalize decimal comma → dot
        obs["Value"] = obs["Value"].str.replace(",", ".", regex=False)
        obs["Value"] = pd.to_numeric(obs["Value"], errors="coerce")

        # Drop duplicates
        before = len(obs)
        obs = obs.drop_duplicates(subset=["WijkenEnBuurten", "Measure"])
        after = len(obs)

        if after < before:
            self.logger.info("Dropped %d duplicates in Observations.", before - after)

        return obs

    # ---- Pivot --------------------------------------------------------------

    def _pivot_observations(self, obs: pd.DataFrame) -> pd.DataFrame:
        """Pivot observations into wide format indexed by WijkenEnBuurten."""
        wide = (
            obs.pivot_table(
                index="WijkenEnBuurten",
                columns="Measure",
                values="Value",
                aggfunc="first"
            )
            .reset_index()
        )
        return wide

    # ---- Load Measure Labels -------------------------------------------------

    def _build_labels(self) -> dict:
        """Build mapping: measure identifier → hierarchical label."""

        # Load MeasureCodes
        measures_path = self.statline_dir / "MeasureCodes.csv"
        groups_path = self.statline_dir / "MeasureGroups.csv"

        if not measures_path.exists() or not groups_path.exists():
            raise FileNotFoundError(
                f"Missing MeasureCodes or MeasureGroups in {self.statline_dir}"
            )

        measures = pd.read_csv(
            measures_path, sep=";", encoding="utf-8-sig", dtype=str
        )[["Identifier", "Title", "MeasureGroupId"]]

        groups = pd.read_csv(
            groups_path, sep=";", encoding="utf-8-sig", dtype=str
        )[["Id", "Title", "ParentId"]].rename(
            columns={"Id": "MeasureGroupId", "Title": "GroupTitle"}
        )

        group_title = dict(zip(groups["MeasureGroupId"], groups["GroupTitle"]))
        parent_id = dict(zip(groups["MeasureGroupId"], groups["ParentId"]))

        def build_label(row: pd.Series) -> str:
            group_id = row["MeasureGroupId"]
            gt = group_title.get(group_id, "")
            parent = parent_id.get(group_id)
            pt = group_title.get(parent, "") if pd.notna(parent) else ""
            parts = [p for p in (pt, gt, row["Title"]) if p]
            return " - ".join(parts)

        return {
            row["Identifier"]: build_label(row)
            for _, row in measures.iterrows()
        }

    # ---- Merge ---------------------------------------------------------------

    def _merge(self, left: GeoDataFrame, wide: pd.DataFrame) -> GeoDataFrame:
        """Merge wide-format measures into the GeoDataFrame."""
        merged = left.merge(
            wide,
            left_on="buurtcode",
            right_on="WijkenEnBuurten",
            how="left"
        ).drop(columns=["WijkenEnBuurten"])

        return merged

    # ---- Select Output Columns ----------------------------------------------

    def _select_output_columns(self, df: GeoDataFrame) -> GeoDataFrame:
        """Fail-fast selection of the final output columns."""

        desired_cols = [
            "geometry", "buurtnaam", "gemeentenaam", "aantal_inwoners",
            "personen_0_tot_15_jaar", "personen_15_tot_25_jaar",
            "personen_65_jaar_en_ouder", "aantal_huishoudens",
            "personenautos_totaal", "level",
            # Herkomst
            "Bevolking naar herkomst - Herkomstland - Nederland",
            "Bevolking naar herkomst - Herkomstland - Europa (exclusief Nederland)",
            "Bevolking naar herkomst - Geboren in Nederland - Europa (exclusief Nederland)",
            "Bevolking naar herkomst - Geboren buiten Nederland - Europa (exclusief Nederland)",
            "Bevolking naar herkomst - Herkomstland - Buiten Europa",
            "Bevolking naar herkomst - Geboren in Nederland - Buiten Europa",
            "Bevolking naar herkomst - Geboren buiten Nederland - Buiten Europa",
            # Woningtype
            "Wonen - Woningen naar eigendom - Koopwoningen",
            "Woningen naar eigendom - Huurwoningen - Huurwoningen totaal",
            "Wonen - Gemiddelde WOZ-waarde van woningen",
            "Wonen - Woningen naar type - Percentage eengezinswoning",
            "Wonen - Woningen naar type - Percentage meergezinswoning",
            "Wonen - Woningen naar bouwjaar - Bouwjaar vanaf 2000",
            "Wonen - Woningen naar bouwjaar - Bouwjaar voor 2000",
            # Stedelijkheid
            "Stedelijkheid - Mate van stedelijkheid",
            # Gezinssamenstelling
            "Bevolking - Particuliere huishoudens - Huishoudens met kinderen",
            # Onderwijs
            "Onderwijs - Hoogst behaald onderwijsniveau - Basisonderwijs, vmbo, mbo1",
            "Onderwijs - Hoogst behaald onderwijsniveau - Havo, vwo, mbo2-4",
            "Onderwijs - Hoogst behaald onderwijsniveau - Hbo, wo",
            # Arbeid
            "Arbeid - Onderverdeling werkenden - Percentage werknemers",
            "Arbeid - Onderverdeling werkenden - Percentage zelfstandigen",
            "Arbeid - Onderverdeling werkenden - Werknemers met vaste arbeidsrelatie",
            "Arbeid - Onderverdeling werkenden - Werknemers met flexibele arbeidsrelatie",
            "Arbeid - Nettoarbeidsparticipatie",
            # Inkomen
            "Inkomen - Inkomen van personen - Aantal inkomensontvangers",
            "Inkomen - Inkomen van huishoudens - Gem. gestandaardiseerd inkomen van huish",
            "Inkomen - Inkomen van huishoudens - Huish. onder of rond sociaal minimum",
            "Inkomen - Inkomen van huishoudens - Mediaan vermogen van particuliere huish.",
            "Inkomen - Inkomen van personen - 40% personen met laagste inkomen",
            "Inkomen - Inkomen van personen - 20% personen met hoogste inkomen",
            # Sociale ondersteuning
            "Zorg - Wmo-cliënten",
            "Zorg - Wmo-cliënten relatief",
            "Zorg - Jongeren met jeugdzorg in natura",
            "Sociale zekerheid - Personen per soort uitkering; WW",
            "Sociale zekerheid - Personen per soort uitkering; AO",
            "Sociale zekerheid - Personen per soort uitkering; Bijstand",
        ]

        missing = [c for c in desired_cols if c not in df.columns]
        if missing:
            raise ValueError(
                "The merged dataset is missing required measures:\n"
                + "\n".join(f"  - {c}" for c in missing)
            )

        return df[desired_cols].copy()