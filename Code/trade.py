# ======================================================
# Trade Volume Fitting & Forecasting (Clean Version)
#   - Read trade volume series from Excel
#   - Fit a logarithmic regression: y = a0 + a1 * ln(t)
#   - Forecast the next 30 periods with 95% confidence interval
#   - Plot observed data, fitted line, forecast, and interval
#
# Author: Bin Lu
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

# ----------------------------
# 1) Input configuration
# ----------------------------
INPUT_FILE = Path(
    "/Users/lubin/Library/CloudStorage/OneDrive-个人/CBAM-Steel/trade.xlsx"
)
SHEET_NAME = "trade"

# Column name in the sheet (target series)
TARGET_COL = "Steel products covered by CBAM"

# Forecast horizon (number of future periods to predict)
FORECAST_HORIZON = 30

# Plot configuration
Y_LABEL = "Volume (10,000 tons)"
TITLE = "Logarithmic Regression with Forecast"
Y_LIM = (-10, 1500)
Y_TICKS = np.arange(0, 1600, 400)

# The x-axis will be labeled like "2000 + t (t)" by default to mimic your original labels.
BASE_YEAR_FOR_LABEL = 2000  # so t=0 -> 2000, t=1 -> 2001, etc.

# ----------------------------
# 2) Read data
# ----------------------------
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

# Ensure target is numeric; non-numeric entries become NaN
y_series = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()

y = y_series.to_numpy(dtype=float)
n = len(y)

# Time index: 1, 2, 3, ..., n
t = np.arange(1, n + 1, dtype=float)

# ----------------------------
# 3) Fit log regression: y = a0 + a1 * ln(t)
# ----------------------------
X = sm.add_constant(np.log(t))         # design matrix: [1, ln(t)]
model = sm.OLS(y, X).fit()             # OLS regression

a0, a1 = model.params
print(f"✅ Fitted model: y = {a0:.3f} + {a1:.3f} × ln(t)")

# Compute R² using statsmodels output (more standard than manual)
print(f"✅ R² = {model.rsquared:.4f}")

# ----------------------------
# 4) Forecast + 95% interval
# ----------------------------
t_future = np.arange(1, n + FORECAST_HORIZON + 1, dtype=float)
X_future = sm.add_constant(np.log(t_future))

pred = model.get_prediction(X_future)
pred_df = pred.summary_frame(alpha=0.05)  # 95% interval by default

y_pred = pred_df["mean"].to_numpy()
ci_low = pred_df["mean_ci_lower"].to_numpy()
ci_high = pred_df["mean_ci_upper"].to_numpy()

# Split indices for plotting (historical vs future)
t_hist = t
t_fore = t_future[n - 1 :]   # start forecast curve from last observed point

# ----------------------------
# 5) Plot
# ----------------------------
plt.figure(figsize=(8, 5))

# Observed data points (historical only)
plt.scatter(
    t_hist,
    y,
    color="steelblue",
    edgecolor="black",
    s=50,
    label="Observed data",
    zorder=3
)

# 95% confidence interval band (over full range: historical + forecast)
plt.fill_between(
    t_future,
    ci_low,
    ci_high,
    color="lightcoral",
    alpha=0.3,
    label="95% confidence interval",
    zorder=1
)

# Fitted line (historical segment)
plt.plot(
    t_hist,
    y_pred[:n],
    color="darkred",
    lw=3.5,
    label="Fitted (historical)",
    zorder=4
)

# Forecast line (future segment; dashed)
plt.plot(
    t_fore,
    y_pred[n - 1 :],
    color="darkred",
    lw=3.5,
    ls="--",
    label="Forecast (future)",
    zorder=4
)

# Equation text (placed above the max observed value)
eq_text = rf"$y = {a0:.2f} + {a1:.2f}\,\ln(t)$"
plt.text(
    1.5,
    max(y) * 1.05,
    eq_text,
    fontsize=16,
    color="darkred",
    fontweight="bold"
)

# ----------------------------
# 6) Styling
# ----------------------------
plt.title(TITLE, fontsize=18, fontweight="bold", pad=12)
plt.ylabel(Y_LABEL, fontsize=14, fontweight="bold", labelpad=8)

plt.ylim(*Y_LIM)
plt.yticks(Y_TICKS, fontsize=14, fontweight="bold")

# X-axis ticks:
# You previously used ticks like 0,10,20,30,40,50 and labeled "year (t)".
# Here we do the same, but ensure the ticks do not exceed the available range.
x_max = int(np.ceil(t_future.max()))
x_ticks = np.arange(0, max(51, x_max + 1), 10)
x_ticks = x_ticks[x_ticks <= x_max]

x_labels = [f"{BASE_YEAR_FOR_LABEL + i:4d} ({i})" for i in x_ticks]
plt.xticks(x_ticks, x_labels, fontsize=14, fontweight="bold")

# Thicker ticks and borders (publication-style)
plt.tick_params(axis="both", which="major", direction="out", length=8, width=2.5, colors="black")

for spine in plt.gca().spines.values():
    spine.set_linewidth(2.5)

plt.grid(alpha=0.3, linestyle="--", zorder=0)
plt.legend(fontsize=12, frameon=True)
plt.tight_layout()
plt.show()


# ======================================================
# Pig Iron Trade Volume Fitting & Forecasting
#   - Read pig iron series from Excel
#   - Fit logarithmic regression: y = a0 + a1 * ln(t)
#   - Forecast next 30 periods with 95% confidence interval
#   - Plot observed data, fitted line, forecast line, and interval band
#
# Author: Bin Lu
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

# ----------------------------
# 1) Input configuration
# ----------------------------
INPUT_FILE = Path(
    "/Users/lubin/Library/CloudStorage/OneDrive-个人/CBAM-Steel/trade.xlsx"
)
SHEET_NAME = "trade"

# Target column (series to model)
TARGET_COL = "Pig iron/10,000 tons"

# Forecast horizon
FORECAST_HORIZON = 30

# Plot configuration
FIGSIZE = (7, 7)
TITLE = "Logarithmic Regression with Forecast (t = 1, 2, 3, ...)"
Y_LABEL = "Volume (10,000 tons)"
Y_LIM = (-0.2, 2.8)
Y_TICKS = np.arange(0, 2.8, 0.5)

# X-axis labeling convention:
# show ticks like "2000 + t (t)" to match your original style
BASE_YEAR_FOR_LABEL = 2000
X_TICKS = np.arange(0, 51, 10)  # display 0,10,20,30,40,50 if within range


# ----------------------------
# 2) Read data
# ----------------------------
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

# Convert to numeric and drop missing values (protects model fit)
y_series = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()

y = y_series.to_numpy(dtype=float)
n = len(y)

# Time index: t = 1, 2, ..., n
t = np.arange(1, n + 1, dtype=float)


# ----------------------------
# 3) Fit log regression: y = a0 + a1 * ln(t)
# ----------------------------
X = sm.add_constant(np.log(t))  # design matrix = [1, ln(t)]
model = sm.OLS(y, X).fit()

a0, a1 = model.params

# Print full regression summary (optional but useful for reporting)
print(model.summary())

print(f"\n✅ Fitted model: y = {a0:.3f} + {a1:.3f} × ln(t)")
print(f"✅ R² (statsmodels) = {model.rsquared:.4f}")


# ----------------------------
# 4) Forecast + 95% confidence interval
# ----------------------------
t_future = np.arange(1, n + FORECAST_HORIZON + 1, dtype=float)
X_future = sm.add_constant(np.log(t_future))

pred = model.get_prediction(X_future)
pred_df = pred.summary_frame(alpha=0.05)  # 95% interval

y_pred = pred_df["mean"].to_numpy()
ci_low = pred_df["mean_ci_lower"].to_numpy()
ci_high = pred_df["mean_ci_upper"].to_numpy()

# For plotting: forecast starts from last historical point
t_fore = t_future[n - 1 :]


# ----------------------------
# 5) Plot
# ----------------------------
plt.figure(figsize=FIGSIZE)

# Observed data points (historical)
plt.scatter(
    t,
    y,
    color="steelblue",
    edgecolor="black",
    s=50,
    label="Observed data (Pig iron)",
    zorder=3,
)

# 95% confidence interval band over full range (historical + forecast)
plt.fill_between(
    t_future,
    ci_low,
    ci_high,
    color="lightcoral",
    alpha=0.3,
    label="95% confidence interval",
    zorder=1,
)

# Fitted line over historical range
plt.plot(
    t,
    y_pred[:n],
    color="darkred",
    lw=3.5,
    label="Fitted (historical)",
    zorder=4,
)

# Forecast line (dashed) over future range
plt.plot(
    t_fore,
    y_pred[n - 1 :],
    color="darkred",
    lw=3.5,
    ls="--",
    label="Forecast (future)",
    zorder=4,
)

# Equation text annotation
eq_text = rf"$y = {a0:.2f} + {a1:.2f}\,\ln(t)$"
plt.text(
    1.5,
    max(y) * 1.2,
    eq_text,
    fontsize=16,
    color="darkred",
    fontweight="bold",
)


# ----------------------------
# 6) Styling (publication-like)
# ----------------------------
plt.title(TITLE, fontsize=18, fontweight="bold", pad=12)
plt.ylabel(Y_LABEL, fontsize=14, fontweight="bold", labelpad=8)

plt.ylim(*Y_LIM)
plt.yticks(Y_TICKS, fontsize=14, fontweight="bold")

# X-axis ticks: keep only those within the available range
x_max = int(np.ceil(t_future.max()))
x_ticks = X_TICKS[X_TICKS <= x_max]
x_labels = [f"{BASE_YEAR_FOR_LABEL + i:4d} ({i})" for i in x_ticks]
plt.xticks(x_ticks, x_labels, fontsize=14, fontweight="bold")

# Thicker tick marks and borders
plt.tick_params(axis="both", which="major", direction="out", length=8, width=2.5, colors="black")

for spine in plt.gca().spines.values():
    spine.set_linewidth(2.5)

plt.grid(alpha=0.3, linestyle="--", zorder=0)
plt.legend(fontsize=12, frameon=True)
plt.tight_layout()
plt.show()

# ======================================================
# Ferroalloys Trade Volume Fitting & Forecasting
#   - Read ferroalloys series from Excel
#   - Fit logarithmic regression: y = a0 + a1 * ln(t)
#   - Forecast next 30 periods with 95% confidence interval
#   - Clip negative predictions to 0 (volume cannot be negative)
#   - Plot observed data, fitted line, forecast line, and interval band
#
# Author: Bin Lu
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

# ----------------------------
# 1) Input configuration
# ----------------------------
INPUT_FILE = Path(
    "/Users/lubin/Library/CloudStorage/OneDrive-个人/CBAM-Steel/trade.xlsx"
)
SHEET_NAME = "trade"

# Target column (series to model)
TARGET_COL = "Ferro-manganese, ferro-chrome, ferro-nickel/10,000 tons"

# Forecast horizon (number of future points)
FORECAST_HORIZON = 30

# Plot configuration
FIGSIZE = (7, 7)
TITLE = "Logarithmic Regression with Forecast (t = 1, 2, 3, ...)"
Y_LABEL = "Volume (10,000 tons)"
Y_LIM = (-0.2, 11)
Y_TICKS = np.arange(0, 11, 2)

# X-axis labeling convention (matches your original style)
BASE_YEAR_FOR_LABEL = 2000
X_TICKS = np.arange(0, 51, 10)


# ----------------------------
# 2) Read data
# ----------------------------
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

# Convert target to numeric and drop missing values for robust fitting
y_series = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()

y = y_series.to_numpy(dtype=float)
n = len(y)

# Time index: t = 1, 2, ..., n
t = np.arange(1, n + 1, dtype=float)


# ----------------------------
# 3) Fit log regression: y = a0 + a1 * ln(t)
# ----------------------------
X = sm.add_constant(np.log(t))   # design matrix = [1, ln(t)]
model = sm.OLS(y, X).fit()

a0, a1 = model.params

# Print full regression summary (useful for diagnostics/reporting)
print(model.summary())

print(f"\n✅ Fitted model: y = {a0:.3f} + {a1:.3f} × ln(t)")
print(f"✅ R² (statsmodels) = {model.rsquared:.4f}")


# ----------------------------
# 4) Forecast + 95% confidence interval
# ----------------------------
t_future = np.arange(1, n + FORECAST_HORIZON + 1, dtype=float)
X_future = sm.add_constant(np.log(t_future))

pred = model.get_prediction(X_future)
pred_df = pred.summary_frame(alpha=0.05)  # 95% interval

# Use NumPy arrays for consistent downstream operations
y_pred = pred_df["mean"].to_numpy()
ci_low = pred_df["mean_ci_lower"].to_numpy()
ci_high = pred_df["mean_ci_upper"].to_numpy()

# ----------------------------
# 4.5) Clip negatives to 0 (trade volume cannot be negative)
# ----------------------------
y_pred = np.clip(y_pred, 0, None)
ci_low = np.clip(ci_low, 0, None)
ci_high = np.clip(ci_high, 0, None)

# Forecast line begins from last observed point
t_fore = t_future[n - 1 :]


# ----------------------------
# 5) Plot
# ----------------------------
plt.figure(figsize=FIGSIZE)

# Observed data points
plt.scatter(
    t,
    y,
    color="steelblue",
    edgecolor="black",
    s=50,
    label="Observed data (Ferroalloys)",
    zorder=3,
)

# 95% confidence interval band (historical + forecast)
plt.fill_between(
    t_future,
    ci_low,
    ci_high,
    color="lightcoral",
    alpha=0.3,
    label="95% confidence interval",
    zorder=1,
)

# Fitted curve for historical period
plt.plot(
    t,
    y_pred[:n],
    color="darkred",
    lw=3.5,
    label="Fitted (historical)",
    zorder=4,
)

# Forecast curve for future period (dashed)
plt.plot(
    t_fore,
    y_pred[n - 1 :],
    color="darkred",
    lw=3.5,
    ls="--",
    label="Forecast (future)",
    zorder=4,
)

# Regression equation annotation
eq_text = rf"$y = {a0:.2f} + {a1:.2f}\,\ln(t)$"
plt.text(
    1.5,
    max(y) * 1.25,
    eq_text,
    fontsize=16,
    color="darkred",
    fontweight="bold",
)


# ----------------------------
# 6) Styling (publication-like)
# ----------------------------
plt.title(TITLE, fontsize=18, fontweight="bold", pad=12)
plt.ylabel(Y_LABEL, fontsize=14, fontweight="bold", labelpad=8)

plt.ylim(*Y_LIM)
plt.yticks(Y_TICKS, fontsize=14, fontweight="bold")

# X-axis ticks: keep only those within the available data range
x_max = int(np.ceil(t_future.max()))
x_ticks = X_TICKS[X_TICKS <= x_max]
x_labels = [f"{BASE_YEAR_FOR_LABEL + i:4d} ({i})" for i in x_ticks]
plt.xticks(x_ticks, x_labels, fontsize=14, fontweight="bold")

# Thicker ticks and borders
plt.tick_params(axis="both", which="major", direction="out", length=8, width=2.5, colors="black")
for spine in plt.gca().spines.values():
    spine.set_linewidth(2.5)

plt.grid(alpha=0.3, linestyle="--", zorder=0)
plt.legend(fontsize=12, frameon=True)
plt.tight_layout()
plt.show()


# ======================================================
# Direct Reduced Iron (DRI) Trade Volume Fitting & Forecasting
#   - Read DRI series from Excel
#   - Fit logarithmic regression: y = a0 + a1 * ln(t)
#   - Forecast next 30 periods with 95% confidence interval
#   - Plot observed data, fitted line, forecast line, and interval band
#
# Author: Bin Lu
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

# ----------------------------
# 1) Input configuration
# ----------------------------
INPUT_FILE = Path(
    "/Users/lubin/Library/CloudStorage/OneDrive-个人/CBAM-Steel/trade.xlsx"
)
SHEET_NAME = "trade"

# Target column (series to model)
TARGET_COL = "Direct Reduced Iron/10,000 tons"

# Forecast horizon
FORECAST_HORIZON = 30

# Plot configuration (your original scale)
FIGSIZE = (7, 7)
TITLE = "Logarithmic Regression with Forecast (t = 1, 2, 3, ...)"
Y_LABEL = "Volume (10,000 tons)"
Y_LIM = (-0.001, 0.045)
Y_TICKS = np.arange(0, 0.045, 0.01)

# X-axis labeling convention (matches your original style)
BASE_YEAR_FOR_LABEL = 2000
X_TICKS = np.arange(0, 51, 10)


# ----------------------------
# 2) Read data
# ----------------------------
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

# Convert to numeric and drop NaNs for stable fitting
y_series = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()

y = y_series.to_numpy(dtype=float)
n = len(y)

# Time index: t = 1, 2, ..., n
t = np.arange(1, n + 1, dtype=float)


# ----------------------------
# 3) Fit log regression: y = a0 + a1 * ln(t)
# ----------------------------
X = sm.add_constant(np.log(t))  # design matrix = [1, ln(t)]
model = sm.OLS(y, X).fit()

a0, a1 = model.params

# Print regression summary (optional but useful)
print(model.summary())

print(f"\n✅ Fitted model: y = {a0:.6f} + {a1:.6f} × ln(t)")
print(f"✅ R² (statsmodels) = {model.rsquared:.4f}")


# ----------------------------
# 4) Forecast + 95% confidence interval
# ----------------------------
t_future = np.arange(1, n + FORECAST_HORIZON + 1, dtype=float)
X_future = sm.add_constant(np.log(t_future))

pred = model.get_prediction(X_future)
pred_df = pred.summary_frame(alpha=0.05)

y_pred = pred_df["mean"].to_numpy()
ci_low = pred_df["mean_ci_lower"].to_numpy()
ci_high = pred_df["mean_ci_upper"].to_numpy()

# Forecast line starts from last historical point
t_fore = t_future[n - 1 :]


# ----------------------------
# 5) Plot
# ----------------------------
plt.figure(figsize=FIGSIZE)

# Observed data
plt.scatter(
    t,
    y,
    color="steelblue",
    edgecolor="black",
    s=50,
    label="Observed data (Direct Reduced Iron)",
    zorder=3,
)

# 95% confidence interval band (historical + forecast)
plt.fill_between(
    t_future,
    ci_low,
    ci_high,
    color="lightcoral",
    alpha=0.3,
    label="95% confidence interval",
    zorder=1,
)

# Fitted (historical) curve
plt.plot(
    t,
    y_pred[:n],
    color="darkred",
    lw=3.5,
    label="Fitted (historical)",
    zorder=4,
)

# Forecast (future) curve - dashed
plt.plot(
    t_fore,
    y_pred[n - 1 :],
    color="darkred",
    lw=3.5,
    ls="--",
    label="Forecast (future)",
    zorder=4,
)

# Equation annotation (more decimals for small-scale series)
eq_text = rf"$y = {a0:.4f} + {a1:.4f}\,\ln(t)$"
plt.text(
    1.5,
    max(y) * 1.3,
    eq_text,
    fontsize=16,
    color="darkred",
    fontweight="bold",
)


# ----------------------------
# 6) Styling
# ----------------------------
plt.title(TITLE, fontsize=18, fontweight="bold", pad=12)
plt.ylabel(Y_LABEL, fontsize=14, fontweight="bold", labelpad=8)

plt.ylim(*Y_LIM)
plt.yticks(Y_TICKS, fontsize=14, fontweight="bold")

# X-axis ticks: keep only ticks within the available range
x_max = int(np.ceil(t_future.max()))
x_ticks = X_TICKS[X_TICKS <= x_max]
x_labels = [f"{BASE_YEAR_FOR_LABEL + i:4d} ({i})" for i in x_ticks]
plt.xticks(x_ticks, x_labels, fontsize=14, fontweight="bold")

# Thicker tick marks and borders
plt.tick_params(axis="both", which="major", direction="out", length=8, width=2.5, colors="black")

for spine in plt.gca().spines.values():
    spine.set_linewidth(2.5)

plt.grid(alpha=0.3, linestyle="--", zorder=0)
plt.legend(fontsize=12, frameon=True)
plt.tight_layout()
plt.show()

# ======================================================
# Crude Steel Trade Volume Fitting & Forecasting
#   - Read crude steel series from Excel
#   - Fit logarithmic regression: y = a0 + a1 * ln(t)
#   - Forecast next 30 periods with 95% confidence interval
#   - Plot observed data, fitted line, forecast line, and interval band
#
# Author: Bin Lu
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

# ----------------------------
# 1) Input configuration
# ----------------------------
INPUT_FILE = Path(
    "/Users/lubin/Library/CloudStorage/OneDrive-个人/CBAM-Steel/trade.xlsx"
)
SHEET_NAME = "trade"

# Target column (series to model)
TARGET_COL = "Crude steel/10,000 tons"

# Forecast horizon
FORECAST_HORIZON = 30

# Plot configuration (match your original scale)
FIGSIZE = (7, 7)
TITLE = "Logarithmic Regression with Forecast (t = 1, 2, 3, ...)"
Y_LABEL = "Volume (10,000 tons)"
Y_LIM = (-0.2, 5.2)
Y_TICKS = np.arange(0, 5.2, 1)

# X-axis labeling convention (matches your original style)
BASE_YEAR_FOR_LABEL = 2000
X_TICKS = np.arange(0, 51, 10)


# ----------------------------
# 2) Read data
# ----------------------------
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

# Convert target to numeric and drop missing values (more robust)
y_series = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()

y = y_series.to_numpy(dtype=float)
n = len(y)

# Time index: t = 1, 2, ..., n
t = np.arange(1, n + 1, dtype=float)


# ----------------------------
# 3) Fit log regression: y = a0 + a1 * ln(t)
# ----------------------------
X = sm.add_constant(np.log(t))  # design matrix = [1, ln(t)]
model = sm.OLS(y, X).fit()

a0, a1 = model.params

# Print regression summary (optional)
print(model.summary())

print(f"\n✅ Fitted model: y = {a0:.4f} + {a1:.4f} × ln(t)")
print(f"✅ R² (statsmodels) = {model.rsquared:.4f}")


# ----------------------------
# 4) Forecast + 95% confidence interval
# ----------------------------
t_future = np.arange(1, n + FORECAST_HORIZON + 1, dtype=float)
X_future = sm.add_constant(np.log(t_future))

pred = model.get_prediction(X_future)
pred_df = pred.summary_frame(alpha=0.05)

y_pred = pred_df["mean"].to_numpy()
ci_low = pred_df["mean_ci_lower"].to_numpy()
ci_high = pred_df["mean_ci_upper"].to_numpy()

# Forecast line starts from last observed point
t_fore = t_future[n - 1 :]


# ----------------------------
# 5) Plot
# ----------------------------
plt.figure(figsize=FIGSIZE)

# Observed data points
plt.scatter(
    t,
    y,
    color="steelblue",
    edgecolor="black",
    s=50,
    label="Observed data (Crude steel)",
    zorder=3,
)

# 95% confidence interval band (historical + forecast)
plt.fill_between(
    t_future,
    ci_low,
    ci_high,
    color="lightcoral",
    alpha=0.3,
    label="95% confidence interval",
    zorder=1,
)

# Fitted line (historical)
plt.plot(
    t,
    y_pred[:n],
    color="darkred",
    lw=3.5,
    label="Fitted (historical)",
    zorder=4,
)

# Forecast line (future, dashed)
plt.plot(
    t_fore,
    y_pred[n - 1 :],
    color="darkred",
    lw=3.5,
    ls="--",
    label="Forecast (future)",
    zorder=4,
)

# Equation annotation
eq_text = rf"$y = {a0:.4f} + {a1:.4f}\,\ln(t)$"
plt.text(
    1.0,
    max(y) * 1.17,
    eq_text,
    fontsize=16,
    color="darkred",
    fontweight="bold",
)


# ----------------------------
# 6) Styling
# ----------------------------
plt.title(TITLE, fontsize=18, fontweight="bold", pad=12)
plt.ylabel(Y_LABEL, fontsize=14, fontweight="bold", labelpad=8)

plt.ylim(*Y_LIM)
plt.yticks(Y_TICKS, fontsize=14, fontweight="bold")

# X-axis ticks: keep ticks that fit within data range
x_max = int(np.ceil(t_future.max()))
x_ticks = X_TICKS[X_TICKS <= x_max]
x_labels = [f"{BASE_YEAR_FOR_LABEL + i:4d} ({i})" for i in x_ticks]
plt.xticks(x_ticks, x_labels, fontsize=14, fontweight="bold")

# Thicker tick marks and borders
plt.tick_params(axis="both", which="major", direction="out", length=8, width=2.5, colors="black")
for spine in plt.gca().spines.values():
    spine.set_linewidth(2.5)

plt.grid(alpha=0.3, linestyle="--", zorder=0)
plt.legend(fontsize=12, frameon=True)
plt.tight_layout()
plt.show()



# ======================================================
# Linear Regression Forecast for CBAM-Covered Steel Products (2020–2050)
#   - Read historical trade volume from Excel
#   - Fit linear model: y = a0 + a1 * Year
#   - Predict annually through 2050 with 95% prediction interval
#   - Plot historical fit (solid) and future forecast (dashed)
#
# Author: Bin Lu
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

# ----------------------------
# 1) Input configuration
# ----------------------------
INPUT_FILE = Path(
    "/Users/lubin/Library/CloudStorage/OneDrive-个人/CBAM-Steel/trade.xlsx"
)
SHEET_NAME = "trade"

# Column names in the Excel sheet
YEAR_COL = "year"
TARGET_COL = "Steel products covered by CBAM"

# Forecast end year
FORECAST_END_YEAR = 2050

# Plot configuration
FIGSIZE = (8, 5)
TITLE = "Linear Regression with Forecast (CBAM Steel products)"
X_LABEL = "Year"
Y_LABEL = "Volume (10,000 tons)"


# ----------------------------
# 2) Read data
# ----------------------------
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

# Convert columns to numeric (robust to strings/blanks) and drop missing values
x = pd.to_numeric(df[YEAR_COL], errors="coerce")
y = pd.to_numeric(df[TARGET_COL], errors="coerce")
mask = x.notna() & y.notna()

x = x.loc[mask].astype(float)
y = y.loc[mask].astype(float)

# ----------------------------
# 3) Fit linear regression: y = a0 + a1 * Year
# ----------------------------
X = sm.add_constant(x)          # add intercept
model = sm.OLS(y, X).fit()

a0, a1 = model.params

print(f"✅ Fitted linear model: y = {a0:.3f} + {a1:.3f} × Year")
print(f"✅ R² (statsmodels) = {model.rsquared:.4f}")

# ----------------------------
# 4) Predict through 2050 (with 95% prediction interval)
# ----------------------------
year_min = int(np.floor(x.min()))
year_max = int(np.ceil(x.max()))
year_future = np.arange(year_min, FORECAST_END_YEAR + 1)

X_future = sm.add_constant(year_future)
pred = model.get_prediction(X_future)
pred_df = pred.summary_frame(alpha=0.05)

# Use *prediction interval* (obs_ci_*) if you want uncertainty for individual observations.
# If you want confidence interval for the mean trend, use mean_ci_* instead.
result_df = pd.DataFrame(
    {
        "Year": year_future,
        "Predicted": pred_df["mean"].to_numpy(),
        "PI_Lower_95": pred_df["obs_ci_lower"].to_numpy(),
        "PI_Upper_95": pred_df["obs_ci_upper"].to_numpy(),
        "CI_Lower_95": pred_df["mean_ci_lower"].to_numpy(),
        "CI_Upper_95": pred_df["mean_ci_upper"].to_numpy(),
    }
)

# ----------------------------
# 5) (Optional) Export results
# ----------------------------
# OUT_FILE = INPUT_FILE.with_name("Steel_products_CBAM_linear_fit_2020_2050.xlsx")
# result_df.to_excel(OUT_FILE, index=False)
# print(f"✅ Exported predictions to: {OUT_FILE}")

# ----------------------------
# 6) Plot
# ----------------------------
plt.figure(figsize=FIGSIZE)

# Observed points
plt.scatter(
    x,
    y,
    color="steelblue",
    edgecolor="black",
    s=50,
    label="Observed data",
    zorder=3,
)

# Historical fitted line (solid)
hist_mask = result_df["Year"] <= year_max
plt.plot(
    result_df.loc[hist_mask, "Year"],
    result_df.loc[hist_mask, "Predicted"],
    color="darkred",
    lw=3.5,
    label="Fitted (historical)",
    zorder=4,
)

# Future forecast line (dashed)
fut_mask = result_df["Year"] >= year_max
plt.plot(
    result_df.loc[fut_mask, "Year"],
    result_df.loc[fut_mask, "Predicted"],
    color="darkred",
    lw=3.5,
    ls="--",
    label="Forecast (future)",
    zorder=4,
)

# 95% prediction interval band (more conservative)
plt.fill_between(
    result_df["Year"],
    result_df["PI_Lower_95"],
    result_df["PI_Upper_95"],
    color="lightcoral",
    alpha=0.3,
    label="95% prediction interval",
    zorder=1,
)

# Equation annotation
eq_text = rf"$y = {a0:.1f} + {a1:.2f}\,\mathrm{{Year}}$"
plt.text(
    x.min() + 1,
    max(y) * 1.6,
    eq_text,
    fontsize=16,
    color="black",
    fontweight="bold",
)

# Labels & style
plt.title(TITLE, fontsize=18, fontweight="bold", pad=12)
plt.xlabel(X_LABEL, fontsize=14, fontweight="bold", labelpad=8)
plt.ylabel(Y_LABEL, fontsize=14, fontweight="bold", labelpad=8)

plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")

plt.tick_params(
    axis="both",
    which="major",
    direction="out",
    length=8,
    width=2.5,
    colors="black",
)

for spine in plt.gca().spines.values():
    spine.set_linewidth(2.5)

plt.grid(alpha=0.3, linestyle="--")
plt.legend(fontsize=12, frameon=True)
plt.tight_layout()
plt.show()


# ======================================================
# Power Function Fit & Forecast for CBAM-Covered Steel Products (2000–2050)
#   - Read historical volume series from Excel
#   - Define relative time index: t = Year - Year_min + 1  (avoids ln(2000) scaling issues)
#   - Fit a power model via log-linearization:
#         y = a * t^b    <=>   ln(y) = ln(a) + b * ln(t)
#   - Forecast annually through 2050 and plot:
#       * observed points
#       * fitted curve (historical)
#       * forecast curve (future, dashed)
#       * 95% CI band (on mean scale; mean_ci in log-space, then exponentiated)
#
# Author: Bin Lu
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

# ----------------------------
# 1) Input configuration
# ----------------------------
INPUT_FILE = Path(
    "/Users/lubin/Library/CloudStorage/OneDrive-个人/CBAM-Steel/trade.xlsx"
)
SHEET_NAME = "trade"

YEAR_COL = "year"
TARGET_COL = "Steel products covered by CBAM"

FORECAST_END_YEAR = 2050

# Plot configuration (match your original)
FIGSIZE = (8, 5)
TITLE = "Power Function Fit and Forecast (2000–2050)"
X_LABEL = "Year"
Y_LABEL = "Volume (10,000 tons)"
Y_LIM = (-100, 2200)
Y_TICKS = np.arange(0, 2200, 500)
X_TICKS = np.arange(2000, 2051, 10)


# ----------------------------
# 2) Read and clean data
# ----------------------------
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

years = pd.to_numeric(df[YEAR_COL], errors="coerce")
y = pd.to_numeric(df[TARGET_COL], errors="coerce")

# Keep rows with valid year and y
mask_valid = years.notna() & y.notna()
years = years.loc[mask_valid].astype(int).to_numpy()
y = y.loc[mask_valid].astype(float).to_numpy()

n = len(y)

# Define relative time index t = 1,2,... based on the first year in the data
# This is numerically nicer than using ln(2000) etc.
t = years - years.min() + 1
print(f"✅ Relative time index t range: {t.min()}–{t.max()}  (Year {years.min()} → t=1)")


# ----------------------------
# 3) Fit power model by log-linearization
#    ln(y) = ln(a) + b * ln(t)
# ----------------------------
# Power model requires y > 0 for log(y)
pos_mask = y > 0

t_pos = t[pos_mask]
y_pos = y[pos_mask]

X = sm.add_constant(np.log(t_pos))     # [1, ln(t)]
Y = np.log(y_pos)                     # ln(y)

model = sm.OLS(Y, X).fit()

ln_a, b = model.params
a = np.exp(ln_a)

print(f"✅ Fitted power model: y = {a:.3f} × t^{b:.3f}")
print(f"✅ R² (log-space) = {model.rsquared:.4f}")  # R² for ln(y) regression


# ----------------------------
# 4) Forecast annually through 2050
# ----------------------------
future_years = np.arange(years.min(), FORECAST_END_YEAR + 1)
t_future = future_years - years.min() + 1

X_future = sm.add_constant(np.log(t_future))

pred = model.get_prediction(X_future)
pred_df = pred.summary_frame(alpha=0.05)

# Back-transform to original scale
# NOTE:
# - Exponentiating mean_ci_* gives a CI for the *conditional mean* under lognormal mapping.
# - If you want a *prediction interval* on original scale, you should use obs_ci_* in log space,
#   then exponentiate, but interpret carefully (and it will be wider).
y_pred = np.exp(pred_df["mean"].to_numpy())
ci_low = np.exp(pred_df["mean_ci_lower"].to_numpy())
ci_high = np.exp(pred_df["mean_ci_upper"].to_numpy())


# ----------------------------
# 5) Goodness-of-fit (original scale, historical years)
# ----------------------------
# Compute fitted values on original scale for ALL observed years (including those with y<=0),
# but note: power model is conceptually defined for positive y.
y_hat_hist = a * (t ** b)

# If you want to evaluate R² only on positive y observations:
#   use pos_mask in both y and y_hat_hist.
r2_original = 1 - np.sum((y[pos_mask] - y_hat_hist[pos_mask]) ** 2) / np.sum(
    (y[pos_mask] - np.mean(y[pos_mask])) ** 2
)
print(f"✅ R² (original scale, y>0 only) = {r2_original:.4f}")


# ----------------------------
# 6) Plot
# ----------------------------
plt.figure(figsize=FIGSIZE)

# Observed data points
plt.scatter(
    years,
    y,
    color="steelblue",
    edgecolor="black",
    s=70,
    label="Observed data",
    zorder=3,
)

# Historical fitted curve (solid)
plt.plot(
    years,
    y_pred[:n],
    color="darkred",
    lw=2.5,
    label="Power fit (historical)",
    zorder=4,
)

# Future forecast curve (dashed) — start from the last observed year to make it continuous
plt.plot(
    future_years[n - 1 :],
    y_pred[n - 1 :],
    color="darkred",
    lw=3.5,
    ls="--",
    label="Forecast (future)",
    zorder=4,
)

# 95% CI band (mean CI on original scale)
plt.fill_between(
    future_years,
    ci_low,
    ci_high,
    color="lightcoral",
    alpha=0.3,
    label="95% CI (mean)",
    zorder=1,
)

# Equation annotation
eq_text = rf"$y = {a:.2f}\,t^{{{b:.2f}}}$"
plt.text(
    years.min() + 2,
    max(y) * 1.5,
    eq_text,
    fontsize=16,
    color="black",
    fontweight="bold",
)

# ----------------------------
# 7) Styling
# ----------------------------
plt.title(TITLE, fontsize=16, fontweight="bold", pad=12)
plt.xlabel(X_LABEL, fontsize=14, fontweight="bold", labelpad=8)
plt.ylabel(Y_LABEL, fontsize=14, fontweight="bold", labelpad=8)

plt.ylim(*Y_LIM)
plt.yticks(Y_TICKS, fontsize=14, fontweight="bold")
plt.xticks(X_TICKS, fontsize=14, fontweight="bold")

plt.tick_params(axis="both", which="major", direction="out", length=8, width=2.5, colors="black")
for spine in plt.gca().spines.values():
    spine.set_linewidth(2.5)

plt.grid(alpha=0.3, linestyle="--", zorder=0)
plt.legend(fontsize=12, frameon=True)
plt.tight_layout()
plt.show()
