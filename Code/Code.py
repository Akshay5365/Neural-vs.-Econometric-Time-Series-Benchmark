# ============================================================
# TIME SERIES STATIONARITY DIAGNOSTICS & BENCHMARK FORECASTING
# Dataset: PJME_hourly.csv (PJM East Regional Transmission Org)
# Models: Econometric SARIMAX vs XGBoost vs TensorFlow/Keras LSTM
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Econometrics & Statistical Testing
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# Machine Learning & Deep Learning (TensorFlow / Keras)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ------------------------------------------------------------
# 1. DATA INGESTION & TEMPORAL RESAMPLING
# ------------------------------------------------------------

FILE_NAME = "PJME_hourly.csv"

if not os.path.exists(FILE_NAME):
    raise FileNotFoundError(
        f"'{FILE_NAME}' not found in working directory. Please place PJME_hourly.csv in the script directory."
    )

df = pd.read_csv(FILE_NAME)

# Identify Datetime and Load columns
datetime_col = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()][0]
load_col = [c for c in df.columns if c != datetime_col][0]

df[datetime_col] = pd.to_datetime(df[datetime_col])
df = df.set_index(datetime_col).sort_index()

# Drop daylight-saving duplicate timestamps
df = df[~df.index.duplicated(keep="first")]

# Resample to Daily Mean Load (MW) for long-horizon econometric tractability
daily_series = df[load_col].resample("D").mean().interpolate(method="time")
daily_series.name = "PJME_MW"

print("--- PJM Hourly Energy Dataset Loaded ---")
print(f"Time Range:          {daily_series.index[0].strftime('%Y-%m-%d')} to {daily_series.index[-1].strftime('%Y-%m-%d')}")
print(f"Total Daily Periods: {len(daily_series)} days")
print(f"Mean Demand:         {daily_series.mean():.2f} MW (Std: {daily_series.std():.2f} MW)")

# ------------------------------------------------------------
# 2. STATISTICAL STATIONARITY SUITE
# ------------------------------------------------------------

def run_stationarity_suite(ts, name="Series"):
    print(f"\n==================== Stationarity Suite: {name} ====================")
    clean_ts = ts.dropna()
    
    # 1. Augmented Dickey-Fuller (ADF) Test
    # H0: Series possesses a unit root (Non-Stationary)
    adf_stat, adf_pval, adf_lags, _, adf_crit, _ = adfuller(clean_ts, autolag="AIC")
    print(f"1. ADF Test Statistic:   {adf_stat:8.4f} | p-value: {adf_pval:.4e} | Lags: {adf_lags}")
    print(f"   -> ADF Decision:      {'REJECT H0 (Stationary)' if adf_pval < 0.05 else 'FAIL TO REJECT H0 (Unit Root Present)'}")
    
    # 2. KPSS Test
    # H0: Series is trend/level stationary
    kpss_stat, kpss_pval, kpss_lags, kpss_crit = kpss(clean_ts, regression="c", nlags="auto")
    print(f"2. KPSS Test Statistic:  {kpss_stat:8.4f} | p-value: {kpss_pval:.4e}")
    print(f"   -> KPSS Decision:     {'REJECT H0 (Non-Stationary)' if kpss_pval < 0.05 else 'FAIL TO REJECT H0 (Stationary)'}")
    
    # 3. STL Decomposition (Weekly & Annual Seasonality Strengths)
    stl = STL(clean_ts, period=7).fit()
    var_resid = np.var(stl.resid)
    var_trend_resid = np.var(stl.trend + stl.resid)
    var_season_resid = np.var(stl.seasonal + stl.resid)
    
    f_trend = max(0.0, 1.0 - var_resid / var_trend_resid)
    f_season = max(0.0, 1.0 - var_resid / var_season_resid)
    print(f"3. Trend Strength (F_t):      {f_trend:.4f} ({'Strong' if f_trend > 0.6 else 'Moderate/Weak'})")
    print(f"4. Weekly Seasonality (F_s):  {f_season:.4f} ({'Strong' if f_season > 0.6 else 'Moderate/Weak'})")

# Run diagnostics on raw series
run_stationarity_suite(daily_series, "Raw PJME Daily Load")

# First difference: Delta Y_t = Y_t - Y_{t-1}
diff_series = daily_series.diff().dropna()
run_stationarity_suite(diff_series, "1st Differenced Series (d=1)")

# ------------------------------------------------------------
# 3. OUT-OF-SAMPLE TRAIN / TEST TEMPORAL SPLIT
# ------------------------------------------------------------

# Strict temporal split: First 85% In-Sample Training, Last 15% Unseen Holdout Testing
split_idx = int(len(daily_series) * 0.85)
train_data = daily_series.iloc[:split_idx]
test_data = daily_series.iloc[split_idx:]
test_horizon = len(test_data)

print(f"\nTraining Days: {len(train_data)} ({train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')})")
print(f"Testing Days:  {len(test_data)} ({test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')})")

# ------------------------------------------------------------
# 4. MODEL 1: ECONOMETRIC SARIMAX & RESIDUAL DIAGNOSTICS
# ------------------------------------------------------------

print("\nFitting Econometric SARIMAX(1, 1, 1)x(1, 0, 1, 7)...")
sarima_model = SARIMAX(
    train_data,
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

# Multi-step out-of-sample forecast on holdout index
sarima_preds = sarima_model.forecast(steps=test_horizon)

# In-sample residual diagnostics
sarima_resid = sarima_model.resid
lb_pval = acorr_ljungbox(sarima_resid, lags=[14], return_df=True)["lb_pvalue"].values[0]
arch_pval = het_arch(sarima_resid)[1]

print("--- SARIMAX Residual Diagnostics ---")
print(f"Ljung-Box Test p-val (Lag 14): {lb_pval:.4f} ({'White Noise' if lb_pval > 0.05 else 'Residual AutoCorrelation Detected'})")
print(f"Engle ARCH Effect p-val:       {arch_pval:.4f} ({'Homoskedastic' if arch_pval > 0.05 else 'ARCH Conditional Variance Detected'})")

# ------------------------------------------------------------
# 5. MODEL 2: FEATURE-ENGINEERED XGBOOST REGRESSOR
# ------------------------------------------------------------

def generate_features(ts, lags=[1, 2, 3, 7, 14, 21, 28, 365]):
    df_feat = pd.DataFrame({"target": ts})
    
    # Autoregressive lag features
    for l in lags:
        df_feat[f"lag_{l}"] = df_feat["target"].shift(l)
        
    # Rolling window summary statistics
    df_feat["rolling_mean_7"] = df_feat["target"].shift(1).rolling(7).mean()
    df_feat["rolling_std_7"] = df_feat["target"].shift(1).rolling(7).std()
    df_feat["rolling_mean_30"] = df_feat["target"].shift(1).rolling(30).mean()
    
    # Calendar & Fourier annual seasonality components
    day_of_year = df_feat.index.dayofyear
    df_feat["dayofweek"] = df_feat.index.dayofweek
    df_feat["month"] = df_feat.index.month
    df_feat["sin_annual"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df_feat["cos_annual"] = np.cos(2 * np.pi * day_of_year / 365.25)
    
    return df_feat.dropna()

df_features = generate_features(daily_series)
X_train = df_features.loc[df_features.index < test_data.index[0]].drop(columns=["target"])
y_train = df_features.loc[df_features.index < test_data.index[0], "target"]
X_test = df_features.loc[df_features.index >= test_data.index[0]].drop(columns=["target"])
y_test = df_features.loc[df_features.index >= test_data.index[0], "target"]

xgb_reg = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED
)
xgb_reg.fit(X_train, y_train)
xgb_preds = pd.Series(xgb_reg.predict(X_test), index=X_test.index)

# ------------------------------------------------------------
# 6. MODEL 3: DEEP STACKED LSTM (TENSORFLOW / KERAS)
# ------------------------------------------------------------

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_data.values.reshape(-1, 1))

seq_len = 30  # 30-day temporal sliding window

def build_lstm_sequences(data, length):
    X, y = [], []
    for i in range(len(data) - length):
        X.append(data[i : i + length])
        y.append(data[i + length])
    return np.array(X), np.array(y)

X_train_lstm, y_train_lstm = build_lstm_sequences(scaled_train, seq_len)

# 2-Layer Stacked LSTM Architecture in Keras
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_len, 1), dropout=0.15),
    LSTM(64, return_sequences=False, dropout=0.15),
    Dense(32, activation="relu"),
    Dense(1)
])

lstm_model.compile(
    optimizer=Adam(learning_rate=0.005),
    loss="mean_squared_error"
)

print("\nTraining Keras Stacked LSTM (50 Epochs)...")
lstm_model.fit(
    X_train_lstm,
    y_train_lstm,
    epochs=50,
    batch_size=64,
    verbose=0,
    shuffle=True
)

# Multi-step recursive rollout across unseen holdout test set
rolling_buffer = list(scaled_train[-seq_len:].flatten())
lstm_raw_predictions = []

for _ in range(test_horizon):
    inp_window = np.array(rolling_buffer[-seq_len:]).reshape(1, seq_len, 1)
    step_pred = lstm_model.predict(inp_window, verbose=0)[0, 0]
    lstm_raw_predictions.append(step_pred)
    rolling_buffer.append(step_pred)

lstm_preds = pd.Series(
    scaler.inverse_transform(np.array(lstm_raw_predictions).reshape(-1, 1)).flatten(),
    index=test_data.index
)

# ------------------------------------------------------------
# 7. UNSEEN HOLDOUT PERFORMANCE & ACCURACY BENCHMARK
# ------------------------------------------------------------

common_idx = y_test.index
actual_unseen = y_test.loc[common_idx].values
sarima_test = sarima_preds.loc[common_idx].values
xgb_test = xgb_preds.loc[common_idx].values
lstm_test = lstm_preds.loc[common_idx].values

def calculate_time_series_metrics(y_true, y_pred, y_train_hist):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    accuracy = 100.0 - mape
    
    # Directional Accuracy (MDA)
    actual_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    mda = np.mean(actual_direction == pred_direction) * 100.0
    
    # Mean Absolute Scaled Error (MASE vs In-Sample Naive Random Walk)
    mae_naive = np.mean(np.abs(np.diff(y_train_hist)))
    mase = mae / mae_naive
    
    return {
        "RMSE (MW)": rmse,
        "MAE (MW)": mae,
        "MAPE (%)": mape,
        "Accuracy (%)": accuracy,
        "Directional Acc (%)": mda,
        "MASE": mase
    }

metrics_dict = {
    "SARIMAX": calculate_time_series_metrics(actual_unseen, sarima_test, train_data.values),
    "XGBoost": calculate_time_series_metrics(actual_unseen, xgb_test, train_data.values),
    "Keras LSTM": calculate_time_series_metrics(actual_unseen, lstm_test, train_data.values)
}

df_metrics = pd.DataFrame(metrics_dict).T
print("\n--- Out-of-Sample Performance on Unseen Holdout Data ---")
print(df_metrics.round(3).to_string())

# ------------------------------------------------------------
# 8. DIEBOLD-MARIANO STATISTICAL SUPERIORITY TEST
# ------------------------------------------------------------

def diebold_mariano_test(actual, pred1, pred2, h=1):
    """
    Diebold-Mariano Test with Newey-West Long-Run Variance Correction.
    H0: Forecast accuracy between Model 1 and Model 2 is equal.
    """
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1**2 - e2**2
    d_bar = np.mean(d)
    
    # Long-Run Variance (Newey-West Bartlett kernel)
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        weight = 1.0 - (lag / h)
        cov_lag = np.cov(d[:-lag], d[lag:])[0, 1]
        var_d += 2.0 * weight * cov_lag
        
    dm_stat = d_bar / np.sqrt(var_d / len(d))
    p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_val

print("\n--- Diebold-Mariano Hypothesis Tests ---")
dm_xgb_sarima, p_xgb_sarima = diebold_mariano_test(actual_unseen, sarima_test, xgb_test)
print(f"XGBoost vs SARIMAX:    DM Stat = {dm_xgb_sarima:8.4f} | p-value = {p_xgb_sarima:.4e} ({'Statistically Significant' if p_xgb_sarima < 0.05 else 'No Significant Difference'})")

dm_xgb_lstm, p_xgb_lstm = diebold_mariano_test(actual_unseen, lstm_test, xgb_test)
print(f"XGBoost vs Keras LSTM: DM Stat = {dm_xgb_lstm:8.4f} | p-value = {p_xgb_lstm:.4e} ({'Statistically Significant' if p_xgb_lstm < 0.05 else 'No Significant Difference'})")

# ------------------------------------------------------------
# 9. VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))
plt.plot(common_idx, actual_unseen, label="Actual PJME Load (MW)", color="black", alpha=0.6, lw=1.2)
plt.plot(common_idx, sarima_test, label="SARIMAX Forecast", color="blue", linestyle="--", lw=1.2)
plt.plot(common_idx, xgb_test, label="XGBoost Forecast", color="green", lw=1.5)
plt.plot(common_idx, lstm_test, label="Keras LSTM Forecast", color="red", linestyle=":", lw=1.2)

plt.title("PJM Energy Load: Out-of-Sample Unseen Forecast Comparison", fontsize=12, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Electricity Demand (MW)")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()