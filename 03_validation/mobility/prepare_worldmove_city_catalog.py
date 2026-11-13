"""Prepare the compact city and country-code catalogs used on the server."""

from pathlib import Path
import json
import sys

sys.path.insert(0, r"${WORKSPACE_ROOT}\\.pydeps_analysis")

import pandas as pd
import pycountry


SOURCE = Path(
    r"${URBAN_DATA_ROOT}\\全球城市边界\建筑环境和移动碳排\DrivingFactorAnalysis\data\Nature_Panoramic_Travel_Analysis\Scheme_B_City_TMI_Downscaled_Portfolio_V17\Tables\city_metrics_with_two_tmi_downscalings.csv"
)
OUT = Path(__file__).resolve().parents[1] / "supplementary_fig" / "worldmove_validation"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE, low_memory=False)
    columns = [
        "ID_UC_G0", "city_name", "country", "iso3", "longitude", "latitude",
        "GC_POP_TOT_2024",
    ]
    catalog = frame[columns].drop_duplicates("ID_UC_G0").copy()
    catalog.to_csv(OUT / "global_city_catalog.csv", index=False, encoding="utf-8")

    code_map = {}
    for country in pycountry.countries:
        code_map[country.alpha_2.upper()] = country.alpha_3.upper()
        code_map[country.alpha_3.upper()] = country.alpha_3.upper()
    # WorldMove uses a few common non-ISO aliases.
    code_map.update({"UK": "GBR", "EL": "GRC", "XK": "XKX"})
    (OUT / "country_code_to_iso3.json").write_text(
        json.dumps(code_map, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"cities={len(catalog)} countries={catalog['iso3'].nunique()}")


if __name__ == "__main__":
    main()
