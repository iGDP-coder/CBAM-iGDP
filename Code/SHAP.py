# ======================================================
# Extract CBAM parameters + scenario results from Excel
# and merge them into a single panel (Year × drivers × scenarios)
# (can be used for SHAP / sensitivity analysis later)
#
# Author: Bin Lu
# ======================================================

import pandas as pd
from pathlib import Path

# ----------------------------
# 1) File paths
# ----------------------------
INPUT_FILE = Path(
    "/Users/lubin/Library/CloudStorage/OneDrive/CBAM-Steel/CBAM-live.xlsm"
)
OUTPUT_FILE = Path("/Users/lubin/Library/CloudStorage/OneDrive/CBAM-Steel/CBAM_parameters_merged.xlsx")


# ----------------------------
# 2) Helper function
# ----------------------------
def read_three_arrays(
    file_path: Path,
    sheet_name: str,
    year_row: int,
    arr1_row: int,
    arr2_row: int,
    arr3_row: int,
    start_col: int,
    end_col: int,
) -> pd.DataFrame:
    
    # Read the whole sheet with no header; keep raw layout (including blanks)
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # Convert 1-based Excel row index to 0-based pandas index
    y_idx = year_row - 1
    a1_idx = arr1_row - 1
    a2_idx = arr2_row - 1
    a3_idx = arr3_row - 1

    # Slice columns (inclusive of end_col)
    col_slice = slice(start_col, end_col + 1)

    years = df.iloc[y_idx, col_slice].tolist()
    arr1 = df.iloc[a1_idx, col_slice].tolist()
    arr2 = df.iloc[a2_idx, col_slice].tolist()
    arr3 = df.iloc[a3_idx, col_slice].tolist()

    out = pd.DataFrame(
        {
            "Year": years,
            "Array1": arr1,
            "Array2": arr2,
            "Array3": arr3,
        }
    )
    return out


def validate_year_alignment(base_year: pd.Series, other_year: pd.Series, name: str) -> None:
    """
    Raise an error if the 'Year' vectors are not identical (prevents silent misalignment).
    """
    if not base_year.equals(other_year):
        raise ValueError(
            f"Year mismatch detected in '{name}'. "
            f"Please check the specified row/column ranges for that sheet."
        )


# ----------------------------
# 3) Batch read five parameter blocks
# ----------------------------

# (1) EU ETS carbon price
ets_price = read_three_arrays(
    file_path=INPUT_FILE,
    sheet_name="5.2.1 CBAM-Carbon market",
    year_row=3,
    arr1_row=4,
    arr2_row=5,
    arr3_row=6,
    start_col=3,  # D
    end_col=33,   # AH
)

# (2) China carbon price
china_price = read_three_arrays(
    file_path=INPUT_FILE,
    sheet_name="5.2.1 CBAM-Carbon market",
    year_row=3,
    arr1_row=21,
    arr2_row=22,
    arr3_row=23,
    start_col=3,  # D
    end_col=33,   # AH
)

# (3) EU ETS benchmark
ets_benchmark = read_three_arrays(
    file_path=INPUT_FILE,
    sheet_name="5.2.2 CBAM-机制规则",
    year_row=22,
    arr1_row=30,
    arr2_row=41,
    arr3_row=52,
    start_col=5,  # F
    end_col=35,   # AJ
)

# (4) China benchmark
china_benchmark = read_three_arrays(
    file_path=INPUT_FILE,
    sheet_name="5.3.2 Emissions and intensity",
    year_row=313,
    arr1_row=322,
    arr2_row=333,
    arr3_row=344,
    start_col=5,  # F
    end_col=35,   # AJ
)

# (5) China product carbon intensity
china_intensity = read_three_arrays(
    file_path=INPUT_FILE,
    sheet_name="5.3.2 Emissions and intensity",
    year_row=231,
    arr1_row=240,
    arr2_row=251,
    arr3_row=262,
    start_col=5,  # F
    end_col=35,   # AJ
)

# ----------------------------
# 4) Validate "Year" alignment across blocks
# ----------------------------
base_year = ets_price["Year"]

validate_year_alignment(base_year, china_price["Year"], "China price")
validate_year_alignment(base_year, ets_benchmark["Year"], "ETS benchmark")
validate_year_alignment(base_year, china_benchmark["Year"], "China benchmark")
validate_year_alignment(base_year, china_intensity["Year"], "China intensity")


# ----------------------------
# 5) Merge into one DataFrame
# ----------------------------
merged_df = pd.DataFrame({"Year": base_year})

# Add each driver with consistent column naming
merged_df["ETS_price_Array1"] = ets_price["Array1"]
merged_df["ETS_price_Array2"] = ets_price["Array2"]
merged_df["ETS_price_Array3"] = ets_price["Array3"]

merged_df["China_price_Array1"] = china_price["Array1"]
merged_df["China_price_Array2"] = china_price["Array2"]
merged_df["China_price_Array3"] = china_price["Array3"]

merged_df["ETS_benchmark_Array1"] = ets_benchmark["Array1"]
merged_df["ETS_benchmark_Array2"] = ets_benchmark["Array2"]
merged_df["ETS_benchmark_Array3"] = ets_benchmark["Array3"]

merged_df["China_benchmark_Array1"] = china_benchmark["Array1"]
merged_df["China_benchmark_Array2"] = china_benchmark["Array2"]
merged_df["China_benchmark_Array3"] = china_benchmark["Array3"]

merged_df["China_intensity_Array1"] = china_intensity["Array1"]
merged_df["China_intensity_Array2"] = china_intensity["Array2"]
merged_df["China_intensity_Array3"] = china_intensity["Array3"]

# Optional: enforce numeric conversion where possible (keeps non-numeric as NaN)
# This is helpful if Excel contains strings like '-', 'n/a', or empty cells.
for c in merged_df.columns:
    if c != "Year":
        merged_df[c] = pd.to_numeric(merged_df[c], errors="coerce")


# ----------------------------
# 6) Quick check + export
# ----------------------------
print("✅ Merged parameter table preview:")
print(merged_df.head())

merged_df.to_excel(OUTPUT_FILE, index=False)
print(f"✅ Exported to: {OUTPUT_FILE}")

# ======================================================
# CBAM Scenario Sensitivity Pipeline (clean & annotated)
#   1) Read scenario outcomes (CBAM_cost) from Excel sheet
#   2) Build scenario-parameter combinations (Scenarios 12–27)
#   3) Merge parameters with outcomes into a long panel (Scenario × Year)
#   4) (Optional) Fit XGBoost + compute permutation SHAP importance
#
# Author: Bin Lu
# ======================================================

import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# 0) File paths
# ----------------------------
INPUT_FILE = Path(
    "/Users/lubin/Library/CloudStorage/OneDrive/CBAM-Steel/CBAM-live.xlsm"
)

OUT_SCENARIO_PARAMS = Path("/Users/lubin/Desktop/CBAM_scenarios_12to27.xlsx")
OUT_MERGED_WITH_COST = Path("/Users/lubin/Desktop/CBAM_combined_with_cost.xlsx")


# ======================================================
# Part A. Read scenario outcome table (CBAM_cost by Year)
# ======================================================
def read_scenario_results(
    file_path: Path,
    sheet_name: str = "scenario",
    year_row: int = 2,              # Excel row number (1-based): "D2:AH2"
    scenario_name_col: int = 2,     # Excel column index (0-based): "C" -> 2
    data_start_row: int = 3,        # Excel row number (1-based) where scenarios start (e.g., row 3)
    data_end_row: int = 29,         # Excel row number (1-based) where scenarios end (inclusive)
    data_start_col: int = 3,        # "D" -> 3
    data_end_col: int = 33          # "AH" -> 33
) -> pd.DataFrame:

    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # Convert 1-based Excel row numbers to 0-based pandas index
    y_idx = year_row - 1
    r0 = data_start_row - 1
    r1 = data_end_row - 1

    # Years (header row)
    years = df_raw.loc[y_idx, data_start_col:data_end_col].tolist()

    # Scenario names (one per row)
    scenario_names = df_raw.loc[r0:r1, scenario_name_col].tolist()

    # Scenario results matrix
    results = df_raw.loc[r0:r1, data_start_col:data_end_col].values

    # Build a clean DataFrame
    results_df = pd.DataFrame(results, columns=years)
    results_df.insert(0, "Scenario", scenario_names)

    # Clean Year column names: convert floats like 2020.0 -> 2020
    def _clean_year(x):
        try:
            return int(float(x))
        except Exception:
            return x

    results_df = results_df.rename(columns={c: _clean_year(c) for c in results_df.columns if c != "Scenario"})

    return results_df


# Read scenario outcomes (wide table)
results_df = read_scenario_results(
    file_path=INPUT_FILE,
    sheet_name="scenario",
    year_row=2,
    scenario_name_col=2,
    data_start_row=3,
    data_end_row=29,
    data_start_col=3,
    data_end_col=33
)

print("✅ Scenario outcome table (wide) preview:")
print(results_df.head())


# ======================================================
# Part B. Build scenario-parameter combinations (12–27)
# ======================================================
# Assumption:
# You already created these 5 DataFrames earlier (from your parameter extraction step),
# and each has columns: Year, Array1, Array2, Array3
#
#   ETS_price, China_price, ETS_benchmark, China_benchmark, China_intensity
#
# If you used the earlier cleaned script, just ensure the objects exist here.

required_dfs = {
    "ETS_price": "ETS_price",
    "China_price": "China_price",
    "ETS_benchmark": "ETS_benchmark",
    "China_benchmark": "China_benchmark",
    "China_intensity": "China_intensity",
}

for k, v in required_dfs.items():
    if v not in globals():
        raise NameError(
            f"Missing DataFrame: {v}. Please run the parameter-extraction script first "
            f"to create {', '.join(required_dfs.values())}."
        )

# Ensure year lengths match (prevents silent mismatch)
n_years = len(ETS_price)
if not all(len(df_) == n_years for df_ in [China_price, ETS_benchmark, China_benchmark, China_intensity]):
    raise ValueError("Year length mismatch among the five parameter tables. Please check your extraction ranges.")


def make_scenario(
    name: str,
    ets_sel: int,
    cn_sel: int,
    ets_bm_sel: int,
    cn_bm_sel: int,
    cn_int_sel: int,
) -> pd.DataFrame:
    """
    Build a long table for one scenario: one row per year.

    Parameters
    ----------
    name : str
        Scenario label used in downstream merge.
    ets_sel, cn_sel, ets_bm_sel, cn_bm_sel, cn_int_sel : int
        Which array to select for each driver (1/2/3).
    """
    out = pd.DataFrame(
        {
            "Year": ETS_price["Year"],
            "ETS_price": ETS_price[f"Array{ets_sel}"],
            "China_price": China_price[f"Array{cn_sel}"],
            "ETS_benchmark": ETS_benchmark[f"Array{ets_bm_sel}"],
            "China_benchmark": China_benchmark[f"Array{cn_bm_sel}"],
            "China_intensity": China_intensity[f"Array{cn_int_sel}"],
            "Scenario": name,
        }
    )
    return out


# Define scenario combinations (12–27)
scenario_specs = [
    ("Scenario_12_BAU",            1, 1, 1, 1, 1),
    ("Scenario_13_ETS2",           2, 1, 1, 1, 1),
    ("Scenario_14_ETS3",           3, 1, 1, 1, 1),
    ("Scenario_15_CNprice2",       1, 2, 1, 1, 1),
    ("Scenario_16_CNprice3",       1, 3, 1, 1, 1),
    ("Scenario_17_ETSbm2",         1, 1, 2, 1, 1),
    ("Scenario_18_ETSbm3",         1, 1, 3, 1, 1),
    ("Scenario_19_CNbm2",          1, 1, 1, 2, 1),
    ("Scenario_20_CNbm3",          1, 1, 1, 3, 1),
    ("Scenario_21_CNint2",         1, 1, 1, 1, 2),
    ("Scenario_22_CNint3",         1, 1, 1, 1, 3),
    ("Scenario_24_ETS2_CNint2",    2, 1, 1, 1, 2),
    ("Scenario_25_ETS2_CNint3",    2, 1, 1, 1, 3),
    ("Scenario_26_ETS3_CNint2",    3, 1, 1, 1, 2),
    ("Scenario_27_ETS3_CNint3",    3, 1, 1, 1, 3),
]

scenario_param_list = [make_scenario(*spec) for spec in scenario_specs]
combined_df = pd.concat(scenario_param_list, ignore_index=True)

print("✅ Combined scenario-parameter panel preview:")
print(combined_df.head())

# Export scenario parameter panel (optional but recommended)
combined_df.to_excel(OUT_SCENARIO_PARAMS, index=False)
print(f"✅ Exported scenario parameter table to: {OUT_SCENARIO_PARAMS}")


# ======================================================
# Part C. Merge parameters with scenario outcomes (CBAM_cost)
# ======================================================
# IMPORTANT:
# Your "scenario" sheet might label scenarios as "scenario12/13/..." rather than "Scenario_12_BAU".
# If so, you need a mapping between our Scenario labels and the sheet labels.
scenario_name_map = {
    "Scenario_12_BAU": "scenario12",
    "Scenario_13_ETS2": "scenario13",
    "Scenario_14_ETS3": "scenario14",
    "Scenario_15_CNprice2": "scenario15",
    "Scenario_16_CNprice3": "scenario16",
    "Scenario_17_ETSbm2": "scenario17",
    "Scenario_18_ETSbm3": "scenario18",
    "Scenario_19_CNbm2": "scenario19",
    "Scenario_20_CNbm3": "scenario20",
    "Scenario_21_CNint2": "scenario21",
    "Scenario_22_CNint3": "scenario22",
    "Scenario_24_ETS2_CNint2": "scenario24",
    "Scenario_25_ETS2_CNint3": "scenario25",
    "Scenario_26_ETS3_CNint2": "scenario26",
    "Scenario_27_ETS3_CNint3": "scenario27",
}

# Add a "Scenario_CN" column for matching the outcome sheet names
combined_df["Scenario_CN"] = combined_df["Scenario"].map(scenario_name_map)

# Convert results_df (wide) -> long: one row per (Scenario, Year)
results_long = results_df.melt(
    id_vars=["Scenario"],
    var_name="Year",
    value_name="CBAM_cost",
)

# Ensure Year is int
results_long["Year"] = results_long["Year"].astype(int)

# Merge using:
#   left:  Scenario_CN + Year
#   right: Scenario (Chinese label) + Year
merged_df = pd.merge(
    combined_df,
    results_long,
    left_on=["Scenario_CN", "Year"],
    right_on=["Scenario", "Year"],
    how="left",
)

# Clean duplicate columns after merge
merged_df = merged_df.drop(columns=["Scenario_y"], errors="ignore").rename(columns={"Scenario_x": "Scenario"})

# Quick QA: check missing outcomes
missing_rate = merged_df["CBAM_cost"].isna().mean()
print(f"✅ Merge completed. Missing CBAM_cost share: {missing_rate:.2%}")

# Export merged dataset
merged_df.to_excel(OUT_MERGED_WITH_COST, index=False)
print(f"✅ Exported merged dataset (with CBAM_cost) to: {OUT_MERGED_WITH_COST}")

print("✅ Final merged table preview:")
print(merged_df.head())


# ======================================================
# Part D (Optional). XGBoost + permutation SHAP importance
# ======================================================
# Notes:
# - Permutation SHAP is model-agnostic and avoids TreeExplainer version issues.
# - If you have many rows, permutation SHAP may be slow; consider sampling.
try:
    import shap
    import xgboost as xgb
    import matplotlib.pyplot as plt

    # Feature columns and target
    features = ["ETS_price", "China_price", "ETS_benchmark", "China_benchmark", "China_intensity"]
    target = "CBAM_cost"

    # Drop rows without target to avoid training issues
    df_model = merged_df.dropna(subset=[target]).copy()

    X = df_model[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df_model[target], errors="coerce")

    # Drop any rows with NaNs in features after coercion
    valid_mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    # Train an XGBoost regressor
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y)

    # Permutation SHAP explainer
    explainer = shap.Explainer(model.predict, X, algorithm="permutation")
    shap_values = explainer(X)

    # Bar plot of mean |SHAP|
    shap.plots.bar(shap_values, show=False)
    plt.title("Feature contribution to CBAM certificate cost", fontsize=15)
    plt.tight_layout()
    plt.show()

    # Export SHAP importance table
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    imp_df = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    print("✅ Permutation SHAP importance (mean |SHAP|):")
    print(imp_df)

except ImportError as e:
    print("⚠️ Optional packages missing for SHAP/XGBoost section:", e)
    print("   You can install via: pip install shap xgboost matplotlib")
