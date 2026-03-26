# -*- coding: utf-8 -*-
"""air_ticket_price_forecasting_thesis.ipynb

Original file is located at
    https://colab.research.google.com/drive/1JaDNsGTRLaJytsC7wAh9uhfxUaGOG8Xh
"""

# ============================================================================
# 0. GLOBAL CONFIGURATION & REPRODUCIBILITY
# ============================================================================
import os, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
warnings.filterwarnings('ignore')

# Install missing libraries
#!pip install pmdarima shap

# ── Deterministic seeds (must precede any library that uses RNG) ─────────────
SEED = 42
os.environ['PYTHONHASHSEED']      = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

# ── Statistical / forecasting libraries ──────────────────────────────────────
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm
from scipy import stats as scipy_stats

# ── sklearn ───────────────────────────────────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              mean_absolute_percentage_error, r2_score)
from sklearn.model_selection import TimeSeriesSplit

# ── Keras ─────────────────────────────────────────────────────────────────────
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                      Input, Concatenate, Bidirectional)
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint)
from tensorflow.keras.optimizers import Adam
import joblib

# ── Plot style ────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams.update({'figure.figsize': (14, 6), 'font.size': 11})

print(f"TensorFlow  : {tf.__version__}")
print(f"Pandas      : {pd.__version__}")
print(f"NumPy       : {np.__version__}")
print("Seeds fixed for reproducibility ✓")

# ============================================================================
# 1. HYPERPARAMETERS
# ============================================================================
DATA_PATH   = r'C:\kq-forecasting\bookings.xlsx'
TARGET_COL  = 'price_per_pax'
DATE_COL    = 'Date of issue day'
DEP_COL     = 'Departure date'

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (implicit)

ARIMA_MAX_P = 5
ARIMA_MAX_Q = 5
SEASONAL_M  = 7          # weekly seasonality

SEQUENCE_LENGTH  = 30    # LSTM look-back window (days)
FORECAST_HORIZON = 30    # future forecast horizon (days)

# Standalone LSTM architecture
SA_LSTM_UNITS  = [64, 32]
SA_DROPOUT     = 0.20
SA_LR          = 1e-3
SA_EPOCHS      = 150
SA_BATCH       = 32
SA_PATIENCE    = 20

# Hybrid LSTM architecture (deeper — models more complex residual patterns)
HY_LSTM_UNITS  = [128, 64]
HY_DROPOUT     = 0.25
HY_LR          = 1e-3
HY_EPOCHS      = 150
HY_BATCH       = 32
HY_PATIENCE    = 20

COVID_START = pd.Timestamp('2020-03-01')

# colour palette shared across all plots
PALETTE = {
    'actual'     : '#2d2d2d',
    'naive'      : '#aaaaaa',
    'arima'      : '#2196F3',   # blue
    'lstm'       : '#FF9800',   # orange
    'hybrid'     : '#E91E63',   # pink/red
    'train'      : '#90CAF9',
    'val'        : '#FFCC80',
    'test'       : '#EF9A9A',
}

# ============================================================================
# 2. DATA LOADING & INSPECTION
# ============================================================================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    print(f"Raw shape  : {df.shape}")
    print(f"Columns    : {df.columns.tolist()}")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("DATA OVERVIEW")
    print("=" * 70)
    print(f"Shape : {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    missing = df.isnull().sum()
    miss_df = pd.DataFrame({'Count': missing, 'Pct': missing / len(df) * 100})
    miss_df = miss_df[miss_df['Count'] > 0]
    print("\n--- MISSING VALUES ---")
    print(miss_df if not miss_df.empty else "None ✓")
    n_dup = df.duplicated().sum()
    print(f"\n--- DUPLICATES ---\nDuplicate rows: {n_dup}" +
          (" ✓" if n_dup == 0 else " — will be dropped"))

# ============================================================================
# 3. RAW PREPROCESSING (global, no leak risk)
# ============================================================================
def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage-1 cleaning: type-casting, structural invalidity removal, feature
    derivation.  No scaling, encoding, or outlier-bound computation here —
    those require fit-on-train discipline.
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[DEP_COL]  = pd.to_datetime(df[DEP_COL])

    n0 = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed     : {n0 - len(df)}")

    n1 = len(df)
    df = df.dropna(subset=['Flown CPV', 'Flown seg pax', DATE_COL, DEP_COL])
    print(f"Key-NaN rows removed   : {n1 - len(df)}")

    df = df[df['Flown seg pax'] > 0]
    df = df[df['Flown CPV'] >= 0]

    df['booking_window'] = (df[DEP_COL] - df[DATE_COL]).dt.days
    df = df[df['booking_window'] >= 0]

    # ── Target ───────────────────────────────────────────────────────────────
    df[TARGET_COL] = df['Flown CPV'] / df['Flown seg pax']

    # ── Temporal features (booking date) ─────────────────────────────────────
    for col, attr in [
        ('booking_month', 'month'), ('booking_day_of_week', 'dayofweek'),
        ('booking_quarter', 'quarter'), ('booking_year', 'year')
    ]:
        df[col] = getattr(df[DATE_COL].dt, attr)
    df['is_weekend_booking'] = df['booking_day_of_week'].isin([5, 6]).astype(int)

    # ── Temporal features (departure date) ───────────────────────────────────
    for col, attr in [
        ('departure_month', 'month'), ('departure_day_of_week', 'dayofweek'),
        ('departure_year', 'year')
    ]:
        df[col] = getattr(df[DEP_COL].dt, attr)
    df['is_weekend_departure'] = df['departure_day_of_week'].isin([5, 6]).astype(int)

    # ── Season (Kenya-specific high/low seasons) ─────────────────────────────
    df['is_high_season'] = df['departure_month'].isin({1, 2, 7, 8, 11, 12}).astype(int)

    # ── Booking urgency (ordinal) ─────────────────────────────────────────────
    def _cat(d):
        if d <= 7:   return 'last_minute'
        if d <= 14:  return 'short_advance'
        if d <= 30:  return 'medium_advance'
        if d <= 60:  return 'long_advance'
        return 'very_long_advance'
    df['booking_category'] = df['booking_window'].apply(_cat)

    # ── Cyclical encoding ─────────────────────────────────────────────────────
    # THESIS NOTE: sin/cos encoding preserves the circular continuity of
    # periodic features (month 12 is adjacent to month 1, etc.)
    for col, period in [
        ('booking_month', 12), ('booking_day_of_week', 7),
        ('departure_month', 12), ('departure_day_of_week', 7)
    ]:
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)

    df['booking_window_log'] = np.log1p(df['booking_window'].clip(lower=0))
    df['days_since_start']   = (df[DATE_COL] - df[DATE_COL].min()).dt.days

    print(f"\nAfter preprocessing    : {df.shape[0]:,} rows, {df.shape[1]} cols")
    return df

# ============================================================================
# 4. ENCODING (fit on train only)
# ============================================================================
def fit_encoders(df_train: pd.DataFrame) -> dict:
    """
    THESIS NOTE: Encoding on the full dataset leaks test-set category
    distributions into training (e.g. frequency encoding uses test frequencies).
    All encoders fit here on training data only.
    """
    cat_cols = df_train.select_dtypes(include=['object', 'category']).columns
    cat_cols = [c for c in cat_cols
                if c not in [DATE_COL, DEP_COL, 'booking_category']
                and 'date' not in c.lower()]
    encoders = {}
    for col in cat_cols:
        if df_train[col].nunique() <= 30:
            le = LabelEncoder()
            le.fit(df_train[col].astype(str))
            encoders[col] = ('label', le)
        else:
            freq = df_train[col].value_counts(normalize=True).to_dict()
            encoders[col] = ('freq', freq)
    _ord = {'last_minute': 0, 'short_advance': 1, 'medium_advance': 2,
            'long_advance': 3, 'very_long_advance': 4}
    encoders['booking_category'] = ('ordinal', _ord)
    print(f"Fitted encoders for {len(encoders)} categorical columns.")
    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col, (enc_type, enc) in encoders.items():
        if col not in df.columns:
            continue
        if enc_type == 'label':
            known = set(enc.classes_)
            df[col + '_enc'] = df[col].astype(str).apply(
                lambda x: enc.transform([x])[0] if x in known else -1)
        elif enc_type == 'freq':
            df[col + '_freq'] = df[col].map(enc).fillna(0.0)
        elif enc_type == 'ordinal':
            df[col + '_ord'] = df[col].map(enc).fillna(-1)
    return df

# ============================================================================
# 5. OUTLIER REMOVAL (bounds from train only)
# ============================================================================
def compute_outlier_bounds(series: pd.Series, n_iqr: float = 3.0) -> tuple:
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - n_iqr * IQR, Q3 + n_iqr * IQR


def apply_outlier_filter(df: pd.DataFrame, col: str,
                          lo: float, hi: float) -> pd.DataFrame:
    n0 = len(df)
    df = df[(df[col] >= lo) & (df[col] <= hi)]
    print(f"Outlier removal ({col}): {n0 - len(df)} rows removed")
    return df

# ============================================================================
# 6. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================================
def temporal_split(df: pd.DataFrame, date_col: str,
                   train_r: float = 0.70, val_r: float = 0.15):
    """
    THESIS NOTE: Random splitting of time-series data causes temporal leakage.
    Chronological ordering ensures the model only observes the past during
    training, matching real-world deployment conditions.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n  = len(df)
    n_tr = int(n * train_r)
    n_va = int(n * val_r)
    df_tr = df.iloc[:n_tr].copy()
    df_va = df.iloc[n_tr:n_tr + n_va].copy()
    df_te = df.iloc[n_tr + n_va:].copy()
    print(f"\n{'Split':12s} {'N':>8s}  {'Start':>12s}  {'End':>12s}")
    print("-" * 52)
    for nm, s in [('Train', df_tr), ('Validation', df_va), ('Test', df_te)]:
        print(f"{nm:12s} {len(s):>8,}  "
              f"{s[date_col].min().date()}  {s[date_col].max().date()}")
    return df_tr, df_va, df_te

# ============================================================================
# 7. DAILY AGGREGATION
# ============================================================================
def to_daily_ts(df: pd.DataFrame, target_col: str,
                method: str = 'median') -> pd.DataFrame:
    """
    THESIS NOTE: Aggregation is performed on departure date, not booking date,
    because airline pricing is fundamentally departure-date-driven.
    Forward-fill is applied within each split independently to avoid
    cross-boundary information transfer.
    """
    agg = df.groupby(DEP_COL).agg(
        price=(target_col, method),
        pax=('Flown seg pax', 'sum'),
        avg_bw=('booking_window', 'mean')
    ).reset_index()
    agg = agg.sort_values(DEP_COL).reset_index(drop=True)
    full_range = pd.date_range(agg[DEP_COL].min(), agg[DEP_COL].max(), freq='D')
    agg = agg.set_index(DEP_COL).reindex(full_range)
    agg.index.name = 'date'
    agg = agg.fillna(method='ffill').fillna(method='bfill')
    agg = agg.reset_index()
    print(f"  Daily TS: {len(agg)} days  "
          f"({agg['date'].min().date()} → {agg['date'].max().date()})")
    return agg

# ============================================================================
# 8. STATIONARITY TESTS
# ============================================================================
def stationarity_suite(series: pd.Series, name: str = 'Series') -> dict:
    s = series.dropna()
    adf = adfuller(s, autolag='AIC')
    adf_stat = adf[1] < 0.05
    kp  = kpss(s, regression='c', nlags='auto')
    kp_stat = kp[1] > 0.05
    print(f"\n{'═'*56}")
    print(f"  Stationarity Tests: {name}")
    print(f"{'═'*56}")
    print(f"  ADF   stat={adf[0]:+.4f}  p={adf[1]:.4f}  "
          f"→ {'STATIONARY ✓' if adf_stat else 'NON-STATIONARY ✗'}")
    print(f"  KPSS  stat={kp[0]:+.4f}  p={kp[1]:.4f}  "
          f"→ {'STATIONARY ✓' if kp_stat else 'NON-STATIONARY ✗'}")
    if adf_stat and kp_stat:
        print("  Conclusion: Both tests agree — STATIONARY")
    elif not adf_stat and not kp_stat:
        print("  Conclusion: Both tests agree — NON-STATIONARY; differencing required")
    else:
        print("  Conclusion: Tests disagree — likely TREND-STATIONARY")
    return {'adf': {'stat': adf[0], 'p': adf[1], 'stationary': adf_stat},
            'kpss': {'stat': kp[0],  'p': kp[1],  'stationary': kp_stat}}

# ============================================================================
# 9. EDA PLOTS
# ============================================================================
def plot_price_series(series: pd.Series, title: str = "Price Series") -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    rm = series.rolling(30).mean()
    rs = series.rolling(30).std()
    axes[0].plot(series.index, series.values, alpha=0.6, lw=1, label='Price',
                 color=PALETTE['actual'])
    axes[0].plot(rm.index, rm.values, color='red', lw=2, label='30-Day Mean')
    axes[0].fill_between(series.index, rm - rs, rm + rs,
                         alpha=0.12, color='red', label='±1 SD')
    axes[0].axvline(COVID_START, color='purple', ls='--', lw=1.5,
                    label='COVID-19 Start')
    axes[0].set_title(f'{title} — Rolling Statistics', fontweight='bold')
    axes[0].legend()
    axes[1].hist(series.dropna(), bins=50, edgecolor='white', alpha=0.8)
    axes[1].axvline(series.mean(),   color='red',   ls='--',
                    label=f'Mean: {series.mean():.2f}')
    axes[1].axvline(series.median(), color='green', ls='--',
                    label=f'Median: {series.median():.2f}')
    axes[1].set_title('Price Distribution')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_acf_pacf_panels(series: pd.Series, lags: int = 60,
                          title: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(series.dropna(), lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title(f'ACF — {title}')
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], alpha=0.05, method='ywm')
    axes[1].set_title(f'PACF — {title}')
    plt.tight_layout()
    plt.show()


def plot_seasonal_decomp(series: pd.Series, period: int = 7) -> None:
    try:
        d = seasonal_decompose(series.dropna(), model='additive', period=period)
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        for ax, (lbl, data) in zip(axes, [
            ('Observed', d.observed), ('Trend', d.trend),
            (f'Seasonal (period={period})', d.seasonal), ('Residual', d.resid)
        ]):
            ax.plot(data, lw=1)
            ax.set_title(lbl)
        plt.suptitle('Seasonal Decomposition', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Decomposition error: {e}")

# ============================================================================
# 10. EVALUATION UTILITY & DIEBOLD-MARIANO TEST
# ============================================================================
def evaluate_model(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   label: str = '') -> dict:
    """Compute MAE, RMSE, MAPE, R² and optionally print."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2   = r2_score(y_true, y_pred)
    if label:
        print(f"  {label:28s}  MAE={mae:8.3f}  RMSE={rmse:8.3f}  "
              f"MAPE={mape:6.2f}%  R²={r2:.4f}")
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray,
                          h: int = 1,
                          name1: str = 'M1',
                          name2: str = 'M2') -> dict:
    """
    Diebold-Mariano test for equal predictive accuracy.

    H0: E[d_t] = 0 where d_t = |e1_t|² - |e2_t|²  (squared-error loss)
    Negative DM statistic → e1 < e2 → model 1 is more accurate.

    THESIS NOTE: Raw metric improvements (MAE, RMSE) do not establish
    statistical significance.  The DM test (Harvey, Leybourne & Newbold, 1997)
    provides a formal test of H0: equal forecast accuracy, with a small-sample
    correction for finite test sets.
    """
    e1, e2 = np.array(e1).flatten(), np.array(e2).flatten()
    d = e1 ** 2 - e2 ** 2          # squared-error loss differential
    n = len(d)
    d_bar = d.mean()

    # Newey-West long-run variance (truncated at h-1 lags)
    gamma = [np.mean((d - d_bar) * np.roll(d - d_bar, k)) for k in range(h)]
    nw_var = gamma[0] + 2 * sum(
        (1 - k / (h + 1)) * gamma[k] for k in range(1, h))
    if nw_var <= 0:
        nw_var = gamma[0]          # fallback to simple variance

    dm_stat = d_bar / np.sqrt(nw_var / n)

    # Harvey et al. small-sample correction
    k  = (n + 1 - 2 * h + h * (h - 1) / n) / n
    dm_stat_corr = dm_stat * np.sqrt(k)

    p_value = 2 * scipy_stats.t.sf(np.abs(dm_stat_corr), df=n - 1)

    verdict = (f"{name1} significantly better than {name2}"
               if dm_stat_corr < 0 and p_value < 0.05
               else f"{name2} significantly better than {name1}"
               if dm_stat_corr > 0 and p_value < 0.05
               else "No significant difference")

    print(f"  DM test: {name1} vs {name2}  "
          f"stat={dm_stat_corr:+.4f}  p={p_value:.4f}  → {verdict}")
    return {'dm_stat': dm_stat_corr, 'p_value': p_value, 'verdict': verdict}


def performance_table(results: dict) -> pd.DataFrame:
    """Build, print, and return a formatted performance comparison DataFrame."""
    rows = [{'Model': k, **v} for k, v in results.items()]
    df = pd.DataFrame(rows).set_index('Model')[['MAE', 'RMSE', 'MAPE', 'R2']]
    print("\n" + "=" * 72)
    print("  MODEL PERFORMANCE COMPARISON TABLE (TEST SET)")
    print("=" * 72)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))

    # ── Rank each metric ──────────────────────────────────────────────────────
    rank_df = df.copy()
    for col in ['MAE', 'RMSE', 'MAPE']:
        rank_df[col + '_rank'] = df[col].rank()          # lower is better
    rank_df['R2_rank'] = df['R2'].rank(ascending=False)  # higher is better
    rank_df['mean_rank'] = rank_df[
        ['MAE_rank', 'RMSE_rank', 'MAPE_rank', 'R2_rank']].mean(axis=1)
    print("\n  Rankings (1 = best):")
    print(rank_df[['MAE_rank', 'RMSE_rank', 'MAPE_rank',
                   'R2_rank', 'mean_rank']].to_string(
                       float_format=lambda x: f"{x:.1f}"))
    print("=" * 72)
    return df

# ============================================================================
# 11. NAÏVE SEASONAL BASELINE
# ============================================================================
def naive_seasonal_forecast(train_prices: np.ndarray,
                             n_steps: int,
                             period: int = 7) -> np.ndarray:
    """
    Seasonal naïve: each forecast = last observed value at the same seasonal
    lag (i.e. same day-of-week).  Provides a trivial but theoretically-grounded
    floor benchmark.
    """
    history = list(train_prices)
    preds   = []
    for _ in range(n_steps):
        preds.append(history[-period])
        history.append(history[-period])  # extend history recursively
    return np.array(preds)

# ============================================================================
# 12. ARIMA — AUTO ORDER SELECTION & EVALUATION
# ============================================================================
def fit_auto_arima(train_prices: np.ndarray, seasonal_m: int = 7):
    """
    THESIS NOTE: Stepwise auto_arima (Hyndman & Khandakar, 2008) is more
    computationally efficient than exhaustive grid search and statistically
    principled — d is determined via unit-root tests; AIC guides (p,q) selection.
    Fitted exclusively on training data.
    """
    print("\nRunning Auto-ARIMA on training data …")
    model = pm.auto_arima(
        train_prices,
        start_p=0, max_p=ARIMA_MAX_P,
        start_q=0, max_q=ARIMA_MAX_Q,
        d=None, D=None,
        seasonal=True, m=seasonal_m,
        start_P=0, max_P=2,
        start_Q=0, max_Q=2,
        information_criterion='aic',
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        n_fits=50,
        random_state=SEED
    )
    print(f"\n✅  Best ARIMA order    : {model.order}")
    print(f"    Seasonal order     : {model.seasonal_order}")
    print(f"    AIC                : {model.aic():.2f}")
    return model


def arima_forecast_splits(order: tuple,
                           train_prices: np.ndarray,
                           val_prices: np.ndarray,
                           test_prices: np.ndarray,
                           seasonal_order: tuple = (0, 0, 0, 0)) -> dict:
    """
    Fit SARIMA(p,d,q)(P,D,Q,s) on train → forecast val.
    Refit on train+val → forecast test.
    Extract residuals for each split (used by the hybrid LSTM stage).

    THESIS NOTE: Refitting on train+val before final test evaluation is the
    standard 'fixed-origin' approach; it uses all available labelled data
    while keeping the test set strictly unseen during training.
    Uses SARIMAX (Seasonal ARIMA) to match proposal Section 3.5.1 which
    specifies the full (p,d,q)(P,D,Q,s) parameterisation.
    """
    p, d, q = order
    P, D, Q, s = seasonal_order

    arima_tr = SARIMAX(
        train_prices.astype(float),
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    tr_fitted = arima_tr.fittedvalues
    tr_resid  = train_prices - tr_fitted

    val_fc    = arima_tr.forecast(steps=len(val_prices))
    val_resid = val_prices - val_fc

    tv_prices = np.concatenate([train_prices, val_prices])
    arima_full = SARIMAX(
        tv_prices.astype(float),
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    test_fc    = arima_full.forecast(steps=len(test_prices))
    test_resid = test_prices - test_fc

    return {
        'arima_train'  : arima_tr,
        'arima_full'   : arima_full,
        'train_fitted' : tr_fitted,
        'train_resid'  : tr_resid,
        'val_forecast' : val_fc,
        'val_resid'    : val_resid,
        'test_forecast': test_fc,
        'test_resid'   : test_resid,
    }


def arima_diagnostics(model, title: str = "ARIMA") -> None:
    resid = model.resid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(resid); axes[0, 0].axhline(0, color='red', ls='--')
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 1].hist(resid, bins=30, edgecolor='white', density=True)
    mu, sigma = resid.mean(), resid.std()
    x = np.linspace(resid.min(), resid.max(), 200)
    axes[0, 1].plot(x, scipy_stats.norm.pdf(x, mu, sigma), 'r-', lw=2)
    axes[0, 1].set_title('Residual Distribution')
    scipy_stats.probplot(resid, dist='norm', plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    plot_acf(resid, lags=40, ax=axes[1, 1])
    axes[1, 1].set_title('ACF of Residuals')
    plt.suptitle(f'{title} — Residual Diagnostics', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{title}_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.show()
    max_lag = min(30, len(resid) // 2 - 1)
    lb = acorr_ljungbox(resid, lags=[10, 20, max_lag], return_df=True)
    print("\nLjung-Box Test (H0: no autocorrelation):")
    print(lb)
    sw_s, sw_p = scipy_stats.shapiro(resid[:5000])
    print(f"\nShapiro-Wilk (H0: normality): stat={sw_s:.4f}, p={sw_p:.4f}")
    print("→", "Residuals appear normal ✓" if sw_p > 0.05 else
          "Residuals deviate from normality ✗")

# ============================================================================
# 13. WALK-FORWARD CV (all models share the same helper)
# ============================================================================
def walk_forward_cv_arima(prices: np.ndarray, order: tuple,
                           n_splits: int = 5) -> pd.DataFrame:
    """
    THESIS NOTE: Walk-forward (expanding-window) CV mimics real forecasting:
    each fold trains on all past data and evaluates on the immediately
    following period, strictly respecting temporal ordering.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    p, d, q = order
    records = []
    print(f"\n  Walk-Forward CV — ARIMA{order}")
    print("  " + "-" * 52)
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(prices), 1):
        try:
            # Walk-forward CV uses non-seasonal ARIMA for computational speed
            # across folds; the full SARIMAX is used for final evaluation.
            m = ARIMA(prices[tr_idx].astype(float), order=(p, d, q)).fit()
            preds = m.forecast(steps=len(va_idx))
            true  = prices[va_idx]
            rec = {'fold': fold, 'n_train': len(tr_idx), 'n_val': len(va_idx),
                   'MAE' : mean_absolute_error(true, preds),
                   'RMSE': np.sqrt(mean_squared_error(true, preds)),
                   'MAPE': mean_absolute_percentage_error(true, preds) * 100}
            records.append(rec)
            print(f"  Fold {fold}: train={len(tr_idx):4d}  val={len(va_idx):4d}  "
                  f"MAE={rec['MAE']:.3f}  RMSE={rec['RMSE']:.3f}  "
                  f"MAPE={rec['MAPE']:.2f}%")
        except Exception as e:
            print(f"  Fold {fold}: FAILED — {e}")
    cv_df = pd.DataFrame(records)
    if not cv_df.empty:
        print("  CV Summary:")
        for m in ['MAE', 'RMSE', 'MAPE']:
            print(f"    {m:5s}: {cv_df[m].mean():.4f} ± {cv_df[m].std():.4f}")
    return cv_df


def walk_forward_cv_naive(prices: np.ndarray, period: int = 7,
                           n_splits: int = 5) -> pd.DataFrame:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    records = []
    print(f"\n  Walk-Forward CV — Seasonal Naïve (period={period})")
    print("  " + "-" * 52)
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(prices), 1):
        if len(tr_idx) < period:
            print(f"  Fold {fold}: insufficient train data for naïve baseline")
            continue
        preds = naive_seasonal_forecast(prices[tr_idx], len(va_idx), period)
        true  = prices[va_idx]
        rec = {'fold': fold, 'n_train': len(tr_idx), 'n_val': len(va_idx),
               'MAE' : mean_absolute_error(true, preds),
               'RMSE': np.sqrt(mean_squared_error(true, preds)),
               'MAPE': mean_absolute_percentage_error(true, preds) * 100}
        records.append(rec)
        print(f"  Fold {fold}: train={len(tr_idx):4d}  val={len(va_idx):4d}  "
              f"MAE={rec['MAE']:.3f}  RMSE={rec['RMSE']:.3f}  "
              f"MAPE={rec['MAPE']:.2f}%")
    cv_df = pd.DataFrame(records)
    if not cv_df.empty:
        print("  CV Summary:")
        for m in ['MAE', 'RMSE', 'MAPE']:
            print(f"    {m:5s}: {cv_df[m].mean():.4f} ± {cv_df[m].std():.4f}")
    return cv_df

# ============================================================================
# 14A. STANDALONE LSTM — direct price prediction
# ============================================================================
def make_price_sequences(prices: np.ndarray, seq_len: int) -> tuple:
    """
    Convert 1-D price array into windowed (X, y) supervised format.
    X[i] = prices[i : i+seq_len]   →   y[i] = prices[i+seq_len]
    """
    X, y = [], []
    for i in range(len(prices) - seq_len):
        X.append(prices[i:i + seq_len].reshape(-1, 1))
        y.append(prices[i + seq_len])
    return np.array(X), np.array(y)


def build_standalone_lstm(seq_len: int, lstm_units: list,
                           dropout: float, lr: float) -> Sequential:
    """
    Single-input stacked Bidirectional LSTM for direct price forecasting.

    THESIS NOTE: The standalone LSTM is architecturally simpler than the
    hybrid LSTM because it learns from raw (scaled) prices rather than
    structured residuals, which tend to be noisier and higher-frequency.
    A shallower network reduces over-fitting risk on the relatively
    small daily time-series.
    """
    mdl = Sequential(name='Standalone_LSTM')
    mdl.add(Input(shape=(seq_len, 1)))
    mdl.add(Bidirectional(LSTM(lstm_units[0], return_sequences=True,
                               name='bilstm_1')))
    mdl.add(Dropout(dropout))
    mdl.add(BatchNormalization())
    mdl.add(LSTM(lstm_units[1], return_sequences=False, name='lstm_2'))
    mdl.add(Dropout(dropout))
    mdl.add(Dense(32, activation='relu'))
    mdl.add(Dense(1, name='price_output'))
    mdl.compile(optimizer=Adam(learning_rate=lr), loss='huber', metrics=['mae'])
    return mdl


def prepare_standalone_lstm_data(train_p: np.ndarray,
                                  val_p: np.ndarray,
                                  test_p: np.ndarray,
                                  seq_len: int) -> dict:
    """
    Scale prices (fit on train only), build sequences with proper
    boundary continuity (same cross-split prepend technique as hybrid).
    """
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    tr_s  = price_scaler.fit_transform(train_p.reshape(-1, 1)).flatten()
    va_s  = price_scaler.transform(val_p.reshape(-1, 1)).flatten()
    te_s  = price_scaler.transform(test_p.reshape(-1, 1)).flatten()

    # Train sequences
    X_tr, y_tr = make_price_sequences(tr_s, seq_len)

    # Val sequences: prepend last seq_len train points
    va_ctx = np.concatenate([tr_s[-seq_len:], va_s])
    X_va, y_va = make_price_sequences(va_ctx, seq_len)

    # Test sequences: prepend last seq_len val points
    te_ctx = np.concatenate([va_s[-seq_len:], te_s])
    X_te, y_te = make_price_sequences(te_ctx, seq_len)

    print(f"\n  Standalone LSTM sequence shapes:")
    print(f"    Train  X:{X_tr.shape}  y:{y_tr.shape}")
    print(f"    Val    X:{X_va.shape}  y:{y_va.shape}")
    print(f"    Test   X:{X_te.shape}  y:{y_te.shape}")

    return {'price_scaler': price_scaler,
            'X_tr': X_tr, 'y_tr': y_tr,
            'X_va': X_va, 'y_va': y_va,
            'X_te': X_te, 'y_te': y_te}


def train_standalone_lstm(data: dict, seq_len: int,
                           units: list, dropout: float,
                           lr: float, epochs: int,
                           batch: int, patience: int) -> tuple:
    mdl = build_standalone_lstm(seq_len, units, dropout, lr)
    mdl.summary()
    cbs = [
        EarlyStopping(monitor='val_loss', patience=patience,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=patience // 2, min_lr=1e-7, verbose=1),
        ModelCheckpoint('best_standalone_lstm.keras', monitor='val_loss',
                        save_best_only=True, verbose=0)
    ]
    hist = mdl.fit(
        data['X_tr'], data['y_tr'],
        validation_data=(data['X_va'], data['y_va']),
        epochs=epochs, batch_size=batch,
        callbacks=cbs,
        shuffle=False,      # CRITICAL: never shuffle temporal sequences
        verbose=1
    )
    return mdl, hist


def predict_standalone_lstm(mdl, data: dict, split: str = 'te') -> np.ndarray:
    """Run inference and inverse-transform to original price scale."""
    X = data[f'X_{split}']
    preds_s = mdl.predict(X, verbose=0).flatten()
    preds   = data['price_scaler'].inverse_transform(
                  preds_s.reshape(-1, 1)).flatten()
    return preds


def walk_forward_cv_lstm(train_p: np.ndarray, val_p: np.ndarray,
                          seq_len: int, units: list, dropout: float,
                          lr: float, epochs: int, batch: int,
                          patience: int, n_splits: int = 5) -> pd.DataFrame:
    """
    Walk-forward CV for standalone LSTM.

    THESIS NOTE: Each fold retrains the LSTM from scratch on the expanding
    training window.  This is computationally expensive but necessary to
    avoid any form of look-ahead in the CV results.
    """
    all_prices = np.concatenate([train_p, val_p])
    tscv  = TimeSeriesSplit(n_splits=n_splits)
    records = []
    print(f"\n  Walk-Forward CV — Standalone LSTM")
    print("  " + "-" * 52)
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(all_prices), 1):
        if len(tr_idx) <= seq_len:
            print(f"  Fold {fold}: insufficient data (need >{seq_len} points)")
            continue
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        fp = all_prices[tr_idx]
        vp = all_prices[va_idx]

        sc   = MinMaxScaler(feature_range=(0, 1))
        fp_s = sc.fit_transform(fp.reshape(-1, 1)).flatten()
        vp_s = sc.transform(vp.reshape(-1, 1)).flatten()
        va_ctx = np.concatenate([fp_s[-seq_len:], vp_s])

        X_tr_f, y_tr_f = make_price_sequences(fp_s, seq_len)
        X_va_f, y_va_f = make_price_sequences(va_ctx, seq_len)

        if len(X_tr_f) == 0 or len(X_va_f) == 0:
            continue

        m = build_standalone_lstm(seq_len, units, dropout, lr)
        cbs = [EarlyStopping(monitor='val_loss', patience=patience,
                              restore_best_weights=True, verbose=0)]
        m.fit(X_tr_f, y_tr_f,
              validation_data=(X_va_f, y_va_f),
              epochs=epochs, batch_size=batch,
              callbacks=cbs, shuffle=False, verbose=0)

        preds_s = m.predict(X_va_f, verbose=0).flatten()
        preds   = sc.inverse_transform(preds_s.reshape(-1, 1)).flatten()
        true    = vp[-len(preds):]

        rec = {'fold': fold, 'n_train': len(tr_idx), 'n_val': len(va_idx),
               'MAE' : mean_absolute_error(true, preds),
               'RMSE': np.sqrt(mean_squared_error(true, preds)),
               'MAPE': mean_absolute_percentage_error(true, preds) * 100}
        records.append(rec)
        print(f"  Fold {fold}: train={len(tr_idx):4d}  val={len(va_idx):4d}  "
              f"MAE={rec['MAE']:.3f}  RMSE={rec['RMSE']:.3f}  "
              f"MAPE={rec['MAPE']:.2f}%")

        del m  # free GPU memory between folds

    cv_df = pd.DataFrame(records)
    if not cv_df.empty:
        print("  CV Summary:")
        for mt in ['MAE', 'RMSE', 'MAPE']:
            print(f"    {mt:5s}: {cv_df[mt].mean():.4f} ± {cv_df[mt].std():.4f}")
    return cv_df


def plot_lstm_training(history, title: str = "LSTM") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_ep = len(history.history['loss'])
    axes[0].plot(history.history['loss'],     label='Train')
    axes[0].plot(history.history['val_loss'], label='Val')
    axes[0].set_title('Huber Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[1].plot(history.history['mae'],     label='Train')
    axes[1].plot(history.history['val_mae'], label='Val')
    axes[1].set_title('MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    plt.suptitle(f'{title} Training History ({n_ep} epochs)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}_training.png',
                dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# 14B. HYBRID LSTM — residual correction on ARIMA
# ============================================================================
def make_residual_sequences(residuals: np.ndarray, exog: np.ndarray,
                              seq_len: int) -> tuple:
    """
    Windowed sequences for the hybrid residual-correction LSTM.
    Returns X_res [N, seq_len, 1], X_exog [N, seq_len, n_exog], y [N,]
    """
    Xr, Xe, y = [], [], []
    for i in range(seq_len, len(residuals)):
        Xr.append(residuals[i - seq_len:i].reshape(-1, 1))
        Xe.append(exog[i - seq_len:i])
        y.append(residuals[i])
    return np.array(Xr), np.array(Xe), np.array(y)


def prepare_hybrid_lstm_data(arima_results: dict,
                              exog_tr: np.ndarray,
                              exog_va: np.ndarray,
                              exog_te: np.ndarray,
                              seq_len: int) -> dict:
    """
    Scale residuals and exog features (fit on train only),
    then build sequences with cross-split boundary continuity.

    CRITICAL LEAKAGE FIX:
      res_scaler and exog_scaler fit on TRAIN ONLY.
    SEQUENCE FIX:
      Val/test prepend the last seq_len training points so the first
      prediction has a full, historically-accurate look-back window.
    """
    res_sc = MinMaxScaler(feature_range=(-1, 1))
    tr_rs  = res_sc.fit_transform(
                 arima_results['train_resid'].reshape(-1, 1)).flatten()
    va_rs  = res_sc.transform(
                 arima_results['val_resid'].reshape(-1, 1)).flatten()
    te_rs  = res_sc.transform(
                 arima_results['test_resid'].reshape(-1, 1)).flatten()

    ex_sc     = StandardScaler()
    exog_tr_s = ex_sc.fit_transform(exog_tr)
    exog_va_s = ex_sc.transform(exog_va)
    exog_te_s = ex_sc.transform(exog_te)

    Xr_tr, Xe_tr, y_tr = make_residual_sequences(tr_rs,   exog_tr_s, seq_len)

    va_r_ctx  = np.concatenate([tr_rs[-seq_len:],    va_rs])
    va_e_ctx  = np.concatenate([exog_tr_s[-seq_len:], exog_va_s], axis=0)
    Xr_va, Xe_va, y_va = make_residual_sequences(va_r_ctx, va_e_ctx, seq_len)

    te_r_ctx  = np.concatenate([va_rs[-seq_len:],    te_rs])
    te_e_ctx  = np.concatenate([exog_va_s[-seq_len:], exog_te_s], axis=0)
    Xr_te, Xe_te, y_te = make_residual_sequences(te_r_ctx, te_e_ctx, seq_len)

    print(f"\n  Hybrid LSTM sequence shapes:")
    print(f"    Train  Xr:{Xr_tr.shape}  Xe:{Xe_tr.shape}  y:{y_tr.shape}")
    print(f"    Val    Xr:{Xr_va.shape}  Xe:{Xe_va.shape}  y:{y_va.shape}")
    print(f"    Test   Xr:{Xr_te.shape}  Xe:{Xe_te.shape}  y:{y_te.shape}")

    return {'res_scaler': res_sc, 'exog_scaler': ex_sc,
            'Xr_tr': Xr_tr, 'Xe_tr': Xe_tr, 'y_tr': y_tr,
            'Xr_va': Xr_va, 'Xe_va': Xe_va, 'y_va': y_va,
            'Xr_te': Xr_te, 'Xe_te': Xe_te, 'y_te': y_te}


def build_hybrid_lstm(seq_len: int, n_exog: int,
                       lstm_units: list, dropout: float, lr: float) -> Model:
    """
    Dual-input Bidirectional LSTM:
      Branch-1: past ARIMA residuals  [batch, seq_len, 1]
      Branch-2: exogenous features    [batch, seq_len, n_exog]
    Merged → Dense correction output.

    THESIS NOTE: The hybrid formula is
        ŷ_hybrid(t) = ŷ_ARIMA(t) + ε̂_LSTM(t)
    where ε̂_LSTM(t) is the predicted ARIMA residual.  This additive
    correction captures non-linearities the ARIMA cannot model (Zhang, 2003).
    Huber loss is used because residuals share the heavy-tailed property
    of the original price series.
    """
    res_in = Input(shape=(seq_len, 1), name='res_input')
    x1 = Bidirectional(LSTM(lstm_units[0], return_sequences=True))(res_in)
    x1 = Dropout(dropout)(x1)
    x1 = BatchNormalization()(x1)
    x1 = LSTM(lstm_units[1], return_sequences=False)(x1)
    x1 = Dropout(dropout)(x1)

    exog_in = Input(shape=(seq_len, n_exog), name='exog_input')
    x2 = LSTM(lstm_units[0], return_sequences=True)(exog_in)
    x2 = Dropout(dropout)(x2)
    x2 = BatchNormalization()(x2)
    x2 = LSTM(lstm_units[1] // 2, return_sequences=False)(x2)
    x2 = Dropout(dropout)(x2)

    merged = Concatenate()([x1, x2])
    x = Dense(64, activation='relu')(merged)
    x = Dropout(dropout / 2)(x)
    x = Dense(32, activation='relu')(x)
    out = Dense(1, name='residual_output')(x)

    mdl = Model(inputs=[res_in, exog_in], outputs=out,
                name='ARIMA_LSTM_Hybrid')
    mdl.compile(optimizer=Adam(learning_rate=lr), loss='huber',
                metrics=['mae'])
    return mdl


def train_hybrid_lstm(data: dict, seq_len: int, n_exog: int,
                       units: list, dropout: float, lr: float,
                       epochs: int, batch: int, patience: int) -> tuple:
    mdl = build_hybrid_lstm(seq_len, n_exog, units, dropout, lr)
    mdl.summary()
    cbs = [
        EarlyStopping(monitor='val_loss', patience=patience,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=patience // 2, min_lr=1e-7, verbose=1),
        ModelCheckpoint('best_hybrid_lstm.keras', monitor='val_loss',
                        save_best_only=True, verbose=0)
    ]
    hist = mdl.fit(
        [data['Xr_tr'], data['Xe_tr']], data['y_tr'],
        validation_data=([data['Xr_va'], data['Xe_va']], data['y_va']),
        epochs=epochs, batch_size=batch,
        callbacks=cbs,
        shuffle=False,
        verbose=1
    )
    return mdl, hist


def apply_hybrid_combination(arima_fc: np.ndarray,
                               mdl, data: dict,
                               res_sc: MinMaxScaler,
                               xr_key: str, xe_key: str) -> tuple:
    """
    Predict LSTM residual correction, inverse-scale, and add to ARIMA forecast.
    Returns (hybrid_predictions, arima_aligned, lstm_corrections).
    Alignment is done by matching the shorter LSTM output to the ARIMA forecast.
    """
    corr_s = mdl.predict([data[xr_key], data[xe_key]], verbose=0).flatten()
    corr   = res_sc.inverse_transform(corr_s.reshape(-1, 1)).flatten()

    n = min(len(arima_fc), len(corr))
    arima_a = arima_fc[-n:]
    corr_a  = corr[-n:]
    assert len(arima_a) == len(corr_a), \
        f"Alignment error: {len(arima_a)} vs {len(corr_a)}"
    return arima_a + corr_a, arima_a, corr_a

# ============================================================================
# 14C. SHAP FEATURE IMPORTANCE — HYBRID LSTM
# ============================================================================
def compute_shap_importance(hybrid_mdl,
                              hy_data: dict,
                              exog_cols: list,
                              n_background: int = 50,
                              n_explain: int = 100) -> pd.DataFrame:
    """
    Compute SHAP values for the exogenous (booking/temporal) input branch of
    the hybrid LSTM to identify the most economically meaningful predictors.

    THESIS NOTE: SHAP (SHapley Additive exPlanations, Lundberg & Lee 2017)
    decomposes each prediction into per-feature additive contributions grounded
    in cooperative game theory.  For neural networks, KernelSHAP uses a
    model-agnostic perturbation approach; DeepSHAP uses back-propagation.
    We use KernelSHAP on the mean-pooled exogenous sequence to keep
    computation tractable while preserving feature-level interpretability.
    This directly satisfies proposal Section 3.8 which requires feature
    importance analysis to verify the model relies on economically
    meaningful predictors (booking window, seasonality, route).

    IMPLEMENTATION NOTE: KernelSHAP operates on 2-D input, so we
    mean-pool the seq_len dimension → shape [N, n_exog] before explaining.
    """
    try:
        import shap
    except ImportError:
        print("  SHAP not installed — run: pip install shap")
        print("  Skipping SHAP analysis.")
        return pd.DataFrame()

    print("\n" + "=" * 60)
    print("SHAP FEATURE IMPORTANCE — HYBRID LSTM (Exog Branch)")
    print("=" * 60)

    # ── 1. Pool sequences to 2-D: [N, n_exog] ────────────────────────────────
    Xe_tr_pool = hy_data['Xe_tr'].mean(axis=1)   # [N_train, n_exog]
    Xe_te_pool = hy_data['Xe_te'].mean(axis=1)   # [N_test,  n_exog]

    # ── 2. Build a wrapper that accepts 2-D exog and uses zero residuals ──────
    # We hold residual input at its training mean to isolate exog contribution.
    Xr_zero = np.zeros((1, hy_data['Xr_tr'].shape[1], 1))

    def model_predict_exog(Xe_2d: np.ndarray) -> np.ndarray:
        """Wrapper: expand pooled 2-D exog back to 3-D for LSTM, predict."""
        n = len(Xe_2d)
        seq_len = hy_data['Xe_tr'].shape[1]
        # Tile the 2-D mean-pooled features across the sequence dimension
        Xe_3d = np.repeat(Xe_2d[:, np.newaxis, :], seq_len, axis=1)
        Xr_rep = np.repeat(Xr_zero, n, axis=0)
        preds = hybrid_mdl.predict([Xr_rep, Xe_3d], verbose=0).flatten()
        return preds

    # ── 3. KernelSHAP with background = random sample of training pool ────────
    bg_idx  = np.random.choice(len(Xe_tr_pool),
                                min(n_background, len(Xe_tr_pool)),
                                replace=False)
    background = Xe_tr_pool[bg_idx]

    ex_idx = np.random.choice(len(Xe_te_pool),
                               min(n_explain, len(Xe_te_pool)),
                               replace=False)
    explain_set = Xe_te_pool[ex_idx]

    print(f"  Background samples : {len(background)}")
    print(f"  Explanation samples: {len(explain_set)}")
    print("  Running KernelSHAP (this may take 1–2 minutes) …")

    explainer  = shap.KernelExplainer(model_predict_exog, background)
    shap_values = explainer.shap_values(explain_set, nsamples=200, silent=True)

    # ── 4. Aggregate: mean |SHAP| per feature ─────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_names = exog_cols if len(exog_cols) == len(mean_abs_shap) \
                    else [f'feature_{i}' for i in range(len(mean_abs_shap))]

    shap_df = pd.DataFrame({
        'feature'         : feature_names,
        'mean_abs_shap'   : mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    shap_df['rank'] = shap_df.index + 1

    print("\n  Top-15 Features by Mean |SHAP| Value:")
    print(shap_df.head(15).to_string(index=False))

    # ── 5. Bar plot ───────────────────────────────────────────────────────────
    top_n = min(20, len(shap_df))
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
    colours = ['#E91E63' if i < 3 else '#2196F3' if i < 10 else '#90CAF9'
               for i in range(top_n)]
    ax.barh(shap_df['feature'][:top_n][::-1],
            shap_df['mean_abs_shap'][:top_n][::-1],
            color=colours[::-1], edgecolor='white')
    ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
    ax.set_title('SHAP Feature Importance — Hybrid LSTM Exogenous Branch\n'
                 '(Top features by mean absolute contribution)',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig('shap_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── 6. SHAP summary plot ──────────────────────────────────────────────────
    try:
        plt.figure(figsize=(10, max(5, top_n * 0.4)))
        shap.summary_plot(shap_values, explain_set,
                          feature_names=feature_names,
                          max_display=top_n, show=False)
        plt.title('SHAP Summary Plot — Hybrid LSTM', fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"  Summary plot skipped: {e}")

    print(f"\n✅  SHAP analysis complete.  Top predictor: "
          f"{shap_df.iloc[0]['feature']} "
          f"(mean |SHAP|={shap_df.iloc[0]['mean_abs_shap']:.5f})")

    return shap_df


def plot_three_way_forecast(daily_ts: pd.DataFrame,
                             n_tr: int, n_va: int,
                             test_true: np.ndarray,
                             arima_pred: np.ndarray,
                             lstm_pred: np.ndarray,
                             hybrid_pred: np.ndarray,
                             naive_pred: np.ndarray,
                             metrics: dict) -> None:
    """
    Four-panel figure showing all models on the test period alongside
    full timeline context.
    """
    test_dates = daily_ts['date'].values[n_tr + n_va:][-len(test_true):]

    fig = plt.figure(figsize=(18, 14))
    gs  = fig.add_gridspec(3, 2, hspace=0.40, wspace=0.30)

    # ── Panel 1 (top, full width): full timeline ──────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(daily_ts['date'][:n_tr], daily_ts['price'][:n_tr],
             color=PALETTE['train'], lw=0.8, label='Train', alpha=0.7)
    ax0.plot(daily_ts['date'][n_tr:n_tr + n_va],
             daily_ts['price'][n_tr:n_tr + n_va],
             color=PALETTE['val'], lw=0.8, label='Validation', alpha=0.7)
    ax0.plot(test_dates, test_true,  color=PALETTE['actual'],  lw=2,
             label='Actual (Test)')
    ax0.plot(test_dates, arima_pred, color=PALETTE['arima'],   lw=1.5,
             ls='--', label=f"ARIMA (MAE={metrics['ARIMA']['MAE']:.2f})")
    ax0.plot(test_dates, lstm_pred,  color=PALETTE['lstm'],    lw=1.5,
             ls='-.',
             label=f"Standalone LSTM (MAE={metrics['Standalone LSTM']['MAE']:.2f})")
    ax0.plot(test_dates, hybrid_pred, color=PALETTE['hybrid'], lw=2,
             label=f"Hybrid (MAE={metrics['ARIMA-LSTM Hybrid']['MAE']:.2f})")
    ax0.axvline(COVID_START, color='purple', ls='--', lw=1.5,
                label='COVID-19 Start')
    ax0.set_title('Full Timeline — All Models', fontweight='bold', fontsize=13)
    ax0.legend(fontsize=9)
    ax0.set_ylabel('Price per Pax (USD)')
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # ── Panel 2 (middle-left): test zoom ARIMA ────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(test_dates, test_true,  color=PALETTE['actual'], lw=2,
             label='Actual')
    ax1.plot(test_dates, arima_pred, color=PALETTE['arima'],  lw=1.5,
             ls='--', label='ARIMA')
    ax1.fill_between(test_dates, test_true, arima_pred,
                     alpha=0.12, color=PALETTE['arima'])
    ax1.set_title(f"Standalone ARIMA  |  "
                  f"MAE={metrics['ARIMA']['MAE']:.2f}  "
                  f"RMSE={metrics['ARIMA']['RMSE']:.2f}",
                  fontsize=10, fontweight='bold')
    ax1.legend(fontsize=9); ax1.set_ylabel('Price per Pax (USD)')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # ── Panel 3 (middle-right): test zoom Standalone LSTM ────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(test_dates, test_true, color=PALETTE['actual'], lw=2,
             label='Actual')
    ax2.plot(test_dates, lstm_pred, color=PALETTE['lstm'],   lw=1.5,
             ls='-.', label='Standalone LSTM')
    ax2.fill_between(test_dates, test_true, lstm_pred,
                     alpha=0.12, color=PALETTE['lstm'])
    ax2.set_title(f"Standalone LSTM  |  "
                  f"MAE={metrics['Standalone LSTM']['MAE']:.2f}  "
                  f"RMSE={metrics['Standalone LSTM']['RMSE']:.2f}",
                  fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9); ax2.set_ylabel('Price per Pax (USD)')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # ── Panel 4 (bottom-left): test zoom Hybrid ───────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(test_dates, test_true,   color=PALETTE['actual'], lw=2,
             label='Actual')
    ax3.plot(test_dates, hybrid_pred, color=PALETTE['hybrid'], lw=1.5,
             label='Hybrid')
    ax3.fill_between(test_dates, test_true, hybrid_pred,
                     alpha=0.12, color=PALETTE['hybrid'])
    ax3.set_title(f"ARIMA-LSTM Hybrid  |  "
                  f"MAE={metrics['ARIMA-LSTM Hybrid']['MAE']:.2f}  "
                  f"RMSE={metrics['ARIMA-LSTM Hybrid']['RMSE']:.2f}",
                  fontsize=10, fontweight='bold')
    ax3.legend(fontsize=9); ax3.set_ylabel('Price per Pax (USD)')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # ── Panel 5 (bottom-right): cumulative error ──────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    for label, pred, col in [
        ('Naïve',           naive_pred,  PALETTE['naive']),
        ('ARIMA',           arima_pred,  PALETTE['arima']),
        ('Standalone LSTM', lstm_pred,   PALETTE['lstm']),
        ('Hybrid',          hybrid_pred, PALETTE['hybrid']),
    ]:
        n = min(len(test_true), len(pred))
        ax4.plot(np.cumsum(np.abs(pred[:n] - test_true[:n])),
                 color=col, lw=1.5, label=label)
    ax4.set_title('Cumulative Absolute Error (Test)', fontsize=10,
                  fontweight='bold')
    ax4.set_xlabel('Test Day'); ax4.set_ylabel('Cumulative |Error|')
    ax4.legend(fontsize=9)

    plt.suptitle('Air Ticket Price Forecasting — Model Comparison',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.savefig('three_way_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_metric_comparison_bars(metrics: dict) -> None:
    """Grouped bar chart: MAE, RMSE, MAPE for all models side-by-side."""
    models   = list(metrics.keys())
    colours  = [PALETTE['naive'], PALETTE['arima'],
                PALETTE['lstm'],  PALETTE['hybrid']]
    metric_labels = ['MAE', 'RMSE', 'MAPE (%)']
    metric_keys   = ['MAE', 'RMSE', 'MAPE']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, mlbl, mkey in zip(axes, metric_labels, metric_keys):
        vals = [metrics[m][mkey] for m in models]
        bars = ax.bar(models, vals, color=colours, edgecolor='white', width=0.5)
        ax.set_title(mlbl, fontweight='bold', fontsize=13)
        ax.set_ylabel(mlbl)
        ax.tick_params(axis='x', rotation=20)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold')
        best_idx = int(np.argmin(vals))
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

    plt.suptitle('Test-Set Error Metrics — All Models\n'
                 '(gold border = best)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('metric_comparison_bars.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_scatter_grid(test_true: np.ndarray,
                      arima_pred: np.ndarray,
                      lstm_pred: np.ndarray,
                      hybrid_pred: np.ndarray,
                      metrics: dict) -> None:
    """Actual vs Predicted scatter plots for all three models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (name, pred, col) in zip(axes, [
        ('ARIMA',           arima_pred,  PALETTE['arima']),
        ('Standalone LSTM', lstm_pred,   PALETTE['lstm']),
        ('ARIMA-LSTM Hybrid', hybrid_pred, PALETTE['hybrid'])
    ]):
        n = min(len(test_true), len(pred))
        ax.scatter(test_true[:n], pred[:n], alpha=0.35, s=15, color=col)
        mn = min(test_true[:n].min(), pred[:n].min())
        mx = max(test_true[:n].max(), pred[:n].max())
        ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5)
        ax.set_title(f'{name}\nR²={metrics[name]["R2"]:.4f}  '
                     f'MAE={metrics[name]["MAE"]:.2f}',
                     fontweight='bold', fontsize=10)
        ax.set_xlabel('Actual Price')
        ax.set_ylabel('Predicted Price')
    plt.suptitle('Actual vs Predicted — All Models',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('scatter_grid.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_cv_summary(cv_results: dict) -> None:
    """Box-plot comparison of walk-forward CV MAE across models."""
    data, labels = [], []
    for model_name, cv_df in cv_results.items():
        if cv_df is not None and not cv_df.empty and 'MAE' in cv_df.columns:
            data.append(cv_df['MAE'].values)
            labels.append(model_name)

    if not data:
        print("No CV data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    colours_list = [PALETTE['naive'], PALETTE['arima'], PALETTE['lstm'],
                    PALETTE['hybrid']]
    for patch, col in zip(bp['boxes'], colours_list[:len(data)]):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_title('Walk-Forward CV — MAE Distribution by Model',
                 fontweight='bold', fontsize=13)
    ax.set_ylabel('MAE (USD)')
    ax.set_xlabel('Model')
    plt.tight_layout()
    plt.savefig('cv_mae_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_error_distributions(test_true: np.ndarray,
                              arima_pred: np.ndarray,
                              lstm_pred: np.ndarray,
                              hybrid_pred: np.ndarray) -> None:
    """Overlapping error histograms for all models."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, pred, col in [
        ('ARIMA',           arima_pred,  PALETTE['arima']),
        ('Standalone LSTM', lstm_pred,   PALETTE['lstm']),
        ('Hybrid',          hybrid_pred, PALETTE['hybrid'])
    ]:
        n = min(len(test_true), len(pred))
        err = test_true[:n] - pred[:n]
        ax.hist(err, bins=30, alpha=0.45, color=col, label=name,
                edgecolor='white')
    ax.axvline(0, color='black', lw=1.5, ls='--')
    ax.set_xlabel('Forecast Error (Actual − Predicted)')
    ax.set_ylabel('Frequency')
    ax.set_title('Forecast Error Distributions — All Models',
                 fontweight='bold', fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig('error_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# 15B. MONTE CARLO DROPOUT — LSTM UNCERTAINTY QUANTIFICATION
# ============================================================================
def mc_dropout_predict(mdl,
                        inputs,
                        n_samples: int = 100) -> tuple:
    """
    Monte Carlo Dropout inference: run the model n_samples times with
    dropout active (training=True) to obtain a distribution of predictions.

    THESIS NOTE: Standard neural network inference uses dropout only during
    training (Srivastava et al., 2014). Gal & Ghahramani (2016) showed that
    running dropout at inference time approximates Bayesian posterior sampling,
    providing calibrated uncertainty estimates without modifying the
    network architecture.  This satisfies proposal Section 3.8 which
    requires prediction interval calibration for LSTM and hybrid models.

    Returns (mean_prediction, std_prediction, lower_95_ci, upper_95_ci)
    """
    preds = []
    for _ in range(n_samples):
        # training=True keeps dropout active
        if isinstance(inputs, list):
            p = mdl(inputs, training=True).numpy().flatten()
        else:
            p = mdl(inputs, training=True).numpy().flatten()
        preds.append(p)
    preds = np.array(preds)          # [n_samples, n_points]
    mean_p = preds.mean(axis=0)
    std_p  = preds.std(axis=0)
    lower  = np.percentile(preds, 2.5,  axis=0)
    upper  = np.percentile(preds, 97.5, axis=0)
    return mean_p, std_p, lower, upper


def mc_dropout_future_forecast(hybrid_mdl,
                                 standalone_mdl,
                                 price_sc: MinMaxScaler,
                                 res_sc: MinMaxScaler,
                                 last_prices_for_sa: np.ndarray,
                                 arima_future: np.ndarray,
                                 arima_test_resid: np.ndarray,
                                 last_exog_raw: np.ndarray,
                                 exog_sc,
                                 horizon: int,
                                 seq_len: int,
                                 n_mc: int = 100) -> dict:
    """
    Generate MC Dropout confidence intervals for Standalone LSTM and Hybrid
    future forecasts.

    THESIS NOTE: Because each MC sample uses a different dropout mask,
    the spread of predictions reflects the model's epistemic uncertainty —
    i.e. how confident the model is about unseen future fares.  Wide
    intervals on specific future dates signal reduced model reliability
    and should prompt analyst review (linked to the dashboard reliability
    heatmap tiers in Section 3.9).
    """
    # ── Standalone LSTM MC Dropout ────────────────────────────────────────────
    hist_s = list(price_sc.transform(
                     last_prices_for_sa[-seq_len:].reshape(-1, 1)).flatten())
    sa_mc_preds = []
    for _ in range(n_mc):
        sa_step_preds = []
        hist_temp = hist_s.copy()
        for _ in range(horizon):
            win = tf.constant(
                np.array(hist_temp[-seq_len:]).reshape(1, seq_len, 1),
                dtype=tf.float32)
            p_s = standalone_mdl(win, training=True).numpy()[0, 0]
            sa_step_preds.append(
                price_sc.inverse_transform([[p_s]])[0, 0])
            hist_temp.append(p_s)
        sa_mc_preds.append(sa_step_preds)

    sa_mc = np.array(sa_mc_preds)        # [n_mc, horizon]
    sa_mean  = sa_mc.mean(axis=0)
    sa_lower = np.percentile(sa_mc, 2.5,  axis=0)
    sa_upper = np.percentile(sa_mc, 97.5, axis=0)

    # ── Hybrid LSTM MC Dropout ────────────────────────────────────────────────
    res_hist  = list(res_sc.transform(
                    arima_test_resid.reshape(-1, 1)).flatten()[-seq_len:])
    exog_hist = list(exog_sc.transform(last_exog_raw)[-seq_len:])
    hy_mc_preds = []
    for _ in range(n_mc):
        corr_step = []
        res_temp  = res_hist.copy()
        exog_temp = exog_hist.copy()
        for _ in range(horizon):
            rw = tf.constant(
                np.array(res_temp[-seq_len:]).reshape(1, seq_len, 1),
                dtype=tf.float32)
            ew = tf.constant(
                np.array(exog_temp[-seq_len:]).reshape(1, seq_len, -1),
                dtype=tf.float32)
            c_s = hybrid_mdl([rw, ew], training=True).numpy()[0, 0]
            corr_step.append(res_sc.inverse_transform([[c_s]])[0, 0])
            res_temp.append(c_s)
            exog_temp.append(exog_temp[-1])
        hy_mc_preds.append(corr_step)

    hy_mc     = np.array(hy_mc_preds)   # [n_mc, horizon]
    hy_corr_mean  = hy_mc.mean(axis=0)
    hy_corr_lower = np.percentile(hy_mc, 2.5,  axis=0)
    hy_corr_upper = np.percentile(hy_mc, 97.5, axis=0)

    hy_mean  = arima_future + hy_corr_mean
    hy_lower = arima_future + hy_corr_lower
    hy_upper = arima_future + hy_corr_upper

    return {
        'sa_mean'  : sa_mean,  'sa_lower' : sa_lower,  'sa_upper' : sa_upper,
        'hy_mean'  : hy_mean,  'hy_lower' : hy_lower,  'hy_upper' : hy_upper,
    }


def recursive_future_forecast(arima_full,
                               hybrid_mdl,
                               res_sc: MinMaxScaler,
                               exog_sc: StandardScaler,
                               last_train_val_prices: np.ndarray,
                               arima_test_resid: np.ndarray,
                               last_exog_raw: np.ndarray,
                               standalone_mdl,
                               price_sc: MinMaxScaler,
                               last_prices_for_sa: np.ndarray,
                               horizon: int,
                               seq_len: int) -> dict:
    """
    Generate horizon-day ahead forecasts for all three models.

    THESIS NOTE: Recursive forecasting uses each step's prediction as the
    input to the next step.  Error accumulates over the horizon — this is
    documented as a limitation; direct multi-output forecasting is suggested
    as an extension.
    """
    # ── ARIMA ─────────────────────────────────────────────────────────────────
    arima_future = arima_full.forecast(steps=horizon)

    # ── Standalone LSTM (recursive, last seq_len prices as seed) ─────────────
    hist_s = list(price_sc.transform(
                      last_prices_for_sa[-seq_len:].reshape(-1, 1)).flatten())
    sa_preds = []
    for _ in range(horizon):
        win = np.array(hist_s[-seq_len:]).reshape(1, seq_len, 1)
        p_s = standalone_mdl.predict(win, verbose=0)[0, 0]
        sa_preds.append(price_sc.inverse_transform([[p_s]])[0, 0])
        hist_s.append(p_s)
    sa_future = np.array(sa_preds)

    # ── Hybrid (recursive LSTM residual correction) ───────────────────────────
    res_hist  = list(res_sc.transform(
                    arima_test_resid.reshape(-1, 1)).flatten()[-seq_len:])
    exog_hist = list(exog_sc.transform(last_exog_raw)[-seq_len:])
    corr_preds = []
    for _ in range(horizon):
        rw = np.array(res_hist[-seq_len:]).reshape(1, seq_len, 1)
        ew = np.array(exog_hist[-seq_len:]).reshape(1, seq_len, -1)
        c_s = hybrid_mdl.predict([rw, ew], verbose=0)[0, 0]
        corr_preds.append(res_sc.inverse_transform([[c_s]])[0, 0])
        res_hist.append(c_s)
        exog_hist.append(exog_hist[-1])
    hybrid_future = arima_future + np.array(corr_preds)

    return {
        'arima_future' : arima_future,
        'sa_lstm_future': sa_future,
        'hybrid_future': hybrid_future
    }

# ============================================================================
# 17. ARTIFACT PERSISTENCE
# ============================================================================
def save_all_artifacts(standalone_mdl, hybrid_mdl,
                        arima_full,
                        price_sc, res_sc, exog_sc,
                        encoders: dict,
                        arima_order: tuple,
                        arima_seasonal_order: tuple,
                        exog_cols: list,
                        forecast_df: pd.DataFrame) -> None:
    standalone_mdl.save('standalone_lstm_model.keras')
    hybrid_mdl.save('hybrid_lstm_model.keras')
    joblib.dump(arima_full,   'arima_model.pkl')
    joblib.dump(price_sc,     'price_scaler.pkl')
    joblib.dump(res_sc,       'residual_scaler.pkl')
    joblib.dump(exog_sc,      'exog_scaler.pkl')
    joblib.dump(encoders,     'label_encoders.pkl')
    joblib.dump({'arima_order': arima_order,
                 'arima_seasonal_order': arima_seasonal_order,
                 'exog_cols': exog_cols,
                 'seq_len': SEQUENCE_LENGTH}, 'model_meta.pkl')
    forecast_df.to_csv('future_forecast_all_models.csv', index=False)
    print("\n✅  Saved artifacts:")
    for f in ['standalone_lstm_model.keras', 'hybrid_lstm_model.keras',
              'arima_model.pkl', 'price_scaler.pkl',
              'residual_scaler.pkl', 'exog_scaler.pkl',
              'label_encoders.pkl', 'model_meta.pkl',
              'future_forecast_all_models.csv']:
        print(f"   {f}")

# ============================================================================
# 18. API-READY INFERENCE CLASS
# ============================================================================
class AirTicketPricePredictor:
    """
    Production inference wrapper for all three models.
    Suitable for a FastAPI / Flask REST endpoint.

    Example (FastAPI):
    ──────────────────
        from fastapi import FastAPI
        from model import AirTicketPricePredictor
        app = FastAPI()
        predictor = AirTicketPricePredictor.load()

        @app.get("/forecast/{horizon}")
        def forecast(horizon: int = 30):
            fc = predictor.forecast_all(horizon=horizon)
            return fc
    """
    def __init__(self, standalone_mdl, hybrid_mdl, arima_model,
                 price_sc, res_sc, exog_sc, encoders, meta: dict):
        self.sa       = standalone_mdl
        self.hybrid   = hybrid_mdl
        self.arima    = arima_model
        self.price_sc = price_sc
        self.res_sc   = res_sc
        self.exog_sc  = exog_sc
        self.encoders = encoders
        self.meta     = meta

    @classmethod
    def load(cls, artifact_dir: str = '.') -> 'AirTicketPricePredictor':
        sa    = tf.keras.models.load_model(
                    os.path.join(artifact_dir, 'standalone_lstm_model.keras'))
        hy    = tf.keras.models.load_model(
                    os.path.join(artifact_dir, 'hybrid_lstm_model.keras'))
        arima = joblib.load(os.path.join(artifact_dir, 'arima_model.pkl'))
        p_sc  = joblib.load(os.path.join(artifact_dir, 'price_scaler.pkl'))
        r_sc  = joblib.load(os.path.join(artifact_dir, 'residual_scaler.pkl'))
        e_sc  = joblib.load(os.path.join(artifact_dir, 'exog_scaler.pkl'))
        encs  = joblib.load(os.path.join(artifact_dir, 'label_encoders.pkl'))
        meta  = joblib.load(os.path.join(artifact_dir, 'model_meta.pkl'))
        return cls(sa, hy, arima, p_sc, r_sc, e_sc, encs, meta)

    def forecast_arima(self, horizon: int = 30) -> np.ndarray:
        return self.arima.forecast(steps=horizon)

    def forecast_all(self, horizon: int = 30) -> dict:
        """Returns a dict with all three model forecasts (simplified)."""
        arima_fc = self.forecast_arima(horizon)
        return {
            'arima':   arima_fc.tolist(),
            'horizon': horizon,
            'note':    'Standalone LSTM and Hybrid require historical '
                       'context — inject via set_context() before calling.'
        }

# ============================================================================
# 19. MAIN PIPELINE
# ============================================================================
def run_pipeline(df_raw: pd.DataFrame) -> dict:
    """
    End-to-end pipeline:
      01. Raw preprocessing
      02. Chronological split
      03. Outlier removal (train bounds)
      04. Categorical encoding (train-fit only)
      05. Daily aggregation per split
      06. EDA + stationarity tests
      07. Auto-ARIMA order selection
      08. Naïve seasonal baseline
      09. Standalone ARIMA evaluation
      10. Standalone LSTM training & evaluation
      11. Hybrid LSTM training & evaluation
      12. Walk-forward CV for all models
      13. Diebold-Mariano significance tests
      14. Comprehensive comparison plots
      15. Future forecast (all models)
      16. Artifact persistence
    """

    # ── 01. Preprocess ───────────────────────────────────────────────────────
    df = preprocess_raw(df_raw)
    inspect_data(df)

    # ── 02. Chronological split ───────────────────────────────────────────────
    df_tr, df_va, df_te = temporal_split(df, DATE_COL, TRAIN_RATIO, VAL_RATIO)

    # ── 03. Outlier removal ───────────────────────────────────────────────────
    lo, hi = compute_outlier_bounds(df_tr[TARGET_COL], n_iqr=3.0)
    print(f"\nOutlier bounds (train IQR×3): [{lo:.2f}, {hi:.2f}]")
    df_tr = apply_outlier_filter(df_tr, TARGET_COL, lo, hi)
    df_va = apply_outlier_filter(df_va, TARGET_COL, lo, hi)
    df_te = apply_outlier_filter(df_te, TARGET_COL, lo, hi)

    # ── 04. Categorical encoding ──────────────────────────────────────────────
    encoders = fit_encoders(df_tr)
    df_tr = apply_encoders(df_tr, encoders)
    df_va = apply_encoders(df_va, encoders)
    df_te = apply_encoders(df_te, encoders)

    # ── 05. Daily aggregation ─────────────────────────────────────────────────
    print("\nAggregating to daily time series …")
    dly_tr = to_daily_ts(df_tr, TARGET_COL)
    dly_va = to_daily_ts(df_va, TARGET_COL)
    dly_te = to_daily_ts(df_te, TARGET_COL)
    daily_ts = pd.concat([dly_tr, dly_va, dly_te]).reset_index(drop=True)
    n_tr, n_va, n_te = len(dly_tr), len(dly_va), len(dly_te)

    tr_p  = dly_tr['price'].values
    va_p  = dly_va['price'].values
    te_p  = dly_te['price'].values

    # ── 06. EDA & stationarity ────────────────────────────────────────────────
    price_ser = pd.Series(tr_p)
    stationarity_suite(price_ser, 'Training Price Series')
    plot_price_series(price_ser, "Training Set Prices")
    plot_acf_pacf_panels(price_ser, lags=60, title="Training Series")
    plot_seasonal_decomp(price_ser, period=SEASONAL_M)

    # ── 07. Auto-ARIMA ────────────────────────────────────────────────────────
    auto_arima   = fit_auto_arima(tr_p, seasonal_m=SEASONAL_M)
    arima_order  = auto_arima.order
    arima_seasonal_order = auto_arima.seasonal_order
    arima_res    = arima_forecast_splits(arima_order, tr_p, va_p, te_p,
                                         seasonal_order=arima_seasonal_order)
    arima_diagnostics(arima_res['arima_train'],
                      title=f"ARIMA{arima_order}")

    # ── 08. Naïve baseline ────────────────────────────────────────────────────
    naive_val  = naive_seasonal_forecast(tr_p,           len(va_p), SEASONAL_M)
    naive_test = naive_seasonal_forecast(
                     np.concatenate([tr_p, va_p]), len(te_p), SEASONAL_M)

    # ── 09. Standalone ARIMA evaluation ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STANDALONE ARIMA — EVALUATION")
    print("=" * 60)
    m_naive_val  = evaluate_model(va_p, naive_val,  'Naïve (Val)')
    m_naive_test = evaluate_model(te_p, naive_test, 'Naïve (Test)')
    m_arima_val  = evaluate_model(va_p, arima_res['val_forecast'],  'ARIMA (Val)')
    m_arima_test = evaluate_model(te_p, arima_res['test_forecast'], 'ARIMA (Test)')

    # ── 10. Standalone LSTM ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STANDALONE LSTM — DATA PREPARATION & TRAINING")
    print("=" * 60)
    sa_data = prepare_standalone_lstm_data(tr_p, va_p, te_p, SEQUENCE_LENGTH)

    sa_mdl, sa_hist = train_standalone_lstm(
        sa_data, SEQUENCE_LENGTH, SA_LSTM_UNITS, SA_DROPOUT,
        SA_LR, SA_EPOCHS, SA_BATCH, SA_PATIENCE)
    plot_lstm_training(sa_hist, "Standalone LSTM")

    # Predictions for val and test
    sa_val_pred  = predict_standalone_lstm(sa_mdl, sa_data, split='va')
    sa_test_pred = predict_standalone_lstm(sa_mdl, sa_data, split='te')

    # Align with price arrays (sequence creation removes first seq_len points)
    n_sa_val  = len(sa_val_pred)
    n_sa_test = len(sa_test_pred)
    sa_val_true  = va_p[-n_sa_val:]
    sa_test_true = te_p[-n_sa_test:]

    print("\nStandalone LSTM — Evaluation:")
    m_sa_val  = evaluate_model(sa_val_true,  sa_val_pred,  'SA-LSTM (Val)')
    m_sa_test = evaluate_model(sa_test_true, sa_test_pred, 'SA-LSTM (Test)')

    # ── 11. Hybrid LSTM ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HYBRID LSTM — DATA PREPARATION & TRAINING")
    print("=" * 60)

    # Build exogenous feature matrix (numeric, aggregated daily)
    exog_exclude = {TARGET_COL, DATE_COL, DEP_COL,
                    'Flown CPV', 'Flown seg pax',
                    'booking_year', 'departure_year'}
    exog_cols = [c for c in df_tr.columns
                 if c not in exog_exclude
                 and df_tr[c].dtype in [np.float64, np.int64,
                                         np.float32, np.int32]
                 and not c.startswith('Unnamed')]
    print(f"  Exogenous features ({len(exog_cols)}): {exog_cols[:8]} …")

    def daily_exog_matrix(df_split, cols):
        agg = df_split.groupby(DEP_COL)[cols].mean().reset_index()
        dr  = pd.date_range(df_split[DEP_COL].min(),
                             df_split[DEP_COL].max(), freq='D')
        agg = agg.set_index(DEP_COL).reindex(dr)
        agg = agg.fillna(method='ffill').fillna(method='bfill')
        return agg[cols].values

    exog_tr_raw = daily_exog_matrix(df_tr, exog_cols)
    exog_va_raw = daily_exog_matrix(df_va, exog_cols)
    exog_te_raw = daily_exog_matrix(df_te, exog_cols)

    # Align lengths to price arrays
    min_tr = min(len(tr_p), len(exog_tr_raw))
    min_va = min(len(va_p), len(exog_va_raw))
    min_te = min(len(te_p), len(exog_te_raw))
    tr_p_a, va_p_a, te_p_a = tr_p[:min_tr], va_p[:min_va], te_p[:min_te]
    exog_tr_raw  = exog_tr_raw[:min_tr]
    exog_va_raw  = exog_va_raw[:min_va]
    exog_te_raw  = exog_te_raw[:min_te]

    # Recompute ARIMA forecasts/residuals on aligned arrays
    arima_res_a = arima_forecast_splits(arima_order, tr_p_a, va_p_a, te_p_a,
                                         seasonal_order=arima_seasonal_order)

    hy_data = prepare_hybrid_lstm_data(
        arima_res_a, exog_tr_raw, exog_va_raw, exog_te_raw, SEQUENCE_LENGTH)

    n_exog    = hy_data['Xe_tr'].shape[2]
    hy_mdl, hy_hist = train_hybrid_lstm(
        hy_data, SEQUENCE_LENGTH, n_exog,
        HY_LSTM_UNITS, HY_DROPOUT, HY_LR,
        HY_EPOCHS, HY_BATCH, HY_PATIENCE)
    plot_lstm_training(hy_hist, "Hybrid LSTM")

    # Combine ARIMA + LSTM corrections
    hy_val_pred, arima_va_aligned, _ = apply_hybrid_combination(
        arima_res_a['val_forecast'],  hy_mdl, hy_data,
        hy_data['res_scaler'], 'Xr_va', 'Xe_va')
    hy_test_pred, arima_te_aligned, _ = apply_hybrid_combination(
        arima_res_a['test_forecast'], hy_mdl, hy_data,
        hy_data['res_scaler'], 'Xr_te', 'Xe_te')

    n_hy_val  = len(hy_val_pred)
    n_hy_test = len(hy_test_pred)
    hy_val_true  = va_p_a[-n_hy_val:]
    hy_test_true = te_p_a[-n_hy_test:]

    print("\nHybrid LSTM — Evaluation:")
    m_hy_val  = evaluate_model(hy_val_true,  hy_val_pred,  'Hybrid (Val)')
    m_hy_test = evaluate_model(hy_test_true, hy_test_pred, 'Hybrid (Test)')

    # ── 11b. SHAP feature importance (Hybrid exog branch) ────────────────────
    print("\n" + "=" * 60)
    print("SHAP FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    shap_df = compute_shap_importance(
        hy_mdl, hy_data, exog_cols,
        n_background=50, n_explain=100)

    # ── 12. Walk-forward CV (all models) ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("WALK-FORWARD CROSS-VALIDATION — ALL MODELS")
    print("=" * 60)

    all_tr_va_prices = np.concatenate([tr_p, va_p])

    cv_naive  = walk_forward_cv_naive(all_tr_va_prices, SEASONAL_M, n_splits=5)
    cv_arima  = walk_forward_cv_arima(all_tr_va_prices, arima_order, n_splits=5)
    cv_sa_lstm = walk_forward_cv_lstm(
        tr_p, va_p, SEQUENCE_LENGTH,
        SA_LSTM_UNITS, SA_DROPOUT, SA_LR,
        SA_EPOCHS, SA_BATCH, SA_PATIENCE, n_splits=5)

    # Note: hybrid walk-forward CV omitted — would require re-fitting ARIMA
    # AND LSTM in each fold, which is computationally prohibitive.
    # The val/test evaluation plus DM tests provide sufficient evidence.
    # (Extending this is flagged as a further-work item.)

    # ── 13. Statistical significance (Diebold-Mariano) ───────────────────────
    print("\n" + "=" * 60)
    print("DIEBOLD-MARIANO SIGNIFICANCE TESTS (TEST SET)")
    print("=" * 60)

    # Align all predictions to the shortest common test array
    min_n = min(len(te_p), len(te_p_a),
                len(naive_test), len(arima_res['test_forecast']),
                len(sa_test_pred), len(hy_test_pred))

    true_dm    = te_p[:min_n]
    err_naive  = true_dm - naive_test[:min_n]
    err_arima  = true_dm - arima_res['test_forecast'][:min_n]
    err_sa     = te_p[-len(sa_test_pred):][:min_n] - sa_test_pred[:min_n]
    err_hybrid = te_p_a[-len(hy_test_pred):][:min_n] - hy_test_pred[:min_n]

    dm_results = {}
    for n1, e1, n2, e2 in [
        ('ARIMA',   err_arima, 'Naïve',  err_naive),
        ('SA-LSTM', err_sa,    'Naïve',  err_naive),
        ('Hybrid',  err_hybrid,'Naïve',  err_naive),
        ('SA-LSTM', err_sa,    'ARIMA',  err_arima),
        ('Hybrid',  err_hybrid,'ARIMA',  err_arima),
        ('Hybrid',  err_hybrid,'SA-LSTM',err_sa),
    ]:
        key = f'{n1} vs {n2}'
        dm_results[key] = diebold_mariano_test(e1, e2, h=1, name1=n1, name2=n2)

    # ── 14. Performance tables & plots ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL PERFORMANCE COMPARISON")
    print("=" * 60)

    # Build unified metrics dict (use the aligned test_true for each model)
    all_metrics = {
        'Naïve (Seasonal)': evaluate_model(te_p[:min_n], naive_test[:min_n]),
        'ARIMA'            : evaluate_model(te_p[:min_n],
                                            arima_res['test_forecast'][:min_n]),
        'Standalone LSTM'  : evaluate_model(
                                 te_p[-len(sa_test_pred):][:min_n],
                                 sa_test_pred[:min_n]),
        'ARIMA-LSTM Hybrid': evaluate_model(
                                 te_p_a[-len(hy_test_pred):][:min_n],
                                 hy_test_pred[:min_n]),
    }
    # Print (and rank) the table
    perf_table = performance_table(all_metrics)

    # Visualisations — align arrays to same base for consistent plots
    true_plot   = te_p[:min_n]
    arima_plot  = arima_res['test_forecast'][:min_n]
    sa_plot     = sa_test_pred[:min_n]
    hy_plot     = hy_test_pred[:min_n]
    naive_plot  = naive_test[:min_n]

    # Rebuild metrics on these aligned arrays for labels inside plots
    plot_metrics = {
        'ARIMA'           : evaluate_model(true_plot, arima_plot),
        'Standalone LSTM' : evaluate_model(true_plot, sa_plot),
        'ARIMA-LSTM Hybrid': evaluate_model(true_plot, hy_plot),
    }

    plot_three_way_forecast(
        daily_ts=daily_ts, n_tr=n_tr, n_va=n_va,
        test_true=true_plot,
        arima_pred=arima_plot, lstm_pred=sa_plot,
        hybrid_pred=hy_plot, naive_pred=naive_plot,
        metrics=plot_metrics)

    plot_metric_comparison_bars(all_metrics)
    plot_scatter_grid(true_plot, arima_plot, sa_plot, hy_plot, plot_metrics)
    plot_error_distributions(true_plot, arima_plot, sa_plot, hy_plot)

    # CV box-plot
    cv_results_all = {
        'Naïve'      : cv_naive,
        'ARIMA'      : cv_arima,
        'SA-LSTM'    : cv_sa_lstm,
    }
    plot_cv_summary(cv_results_all)

    # ── 15. Future forecast ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"FUTURE {FORECAST_HORIZON}-DAY FORECAST — ALL MODELS")
    print("=" * 60)

    fc = recursive_future_forecast(
        arima_full          = arima_res_a['arima_full'],
        hybrid_mdl          = hy_mdl,
        res_sc              = hy_data['res_scaler'],
        exog_sc             = hy_data['exog_scaler'],
        last_train_val_prices = np.concatenate([tr_p_a, va_p_a]),
        arima_test_resid    = arima_res_a['test_resid'],
        last_exog_raw       = exog_te_raw,
        standalone_mdl      = sa_mdl,
        price_sc            = sa_data['price_scaler'],
        last_prices_for_sa  = np.concatenate([va_p, te_p]),
        horizon             = FORECAST_HORIZON,
        seq_len             = SEQUENCE_LENGTH
    )

    # ── MC Dropout uncertainty intervals (proposal Section 3.8) ─────────────
    print("\nComputing Monte Carlo Dropout uncertainty intervals (100 samples) …")
    mc_ci = mc_dropout_future_forecast(
        hybrid_mdl         = hy_mdl,
        standalone_mdl     = sa_mdl,
        price_sc           = sa_data['price_scaler'],
        res_sc             = hy_data['res_scaler'],
        last_prices_for_sa = np.concatenate([va_p, te_p]),
        arima_future       = fc['arima_future'],
        arima_test_resid   = arima_res_a['test_resid'],
        last_exog_raw      = exog_te_raw,
        exog_sc            = hy_data['exog_scaler'],
        horizon            = FORECAST_HORIZON,
        seq_len            = SEQUENCE_LENGTH,
        n_mc               = 100
    )

    last_date    = daily_ts['date'].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1),
                                  periods=FORECAST_HORIZON, freq='D')
    forecast_df  = pd.DataFrame({
        'date'              : future_dates,
        'arima'             : fc['arima_future'],
        'standalone_lstm'   : mc_ci['sa_mean'],
        'standalone_lower95': mc_ci['sa_lower'],
        'standalone_upper95': mc_ci['sa_upper'],
        'hybrid'            : mc_ci['hy_mean'],
        'hybrid_lower95'    : mc_ci['hy_lower'],
        'hybrid_upper95'    : mc_ci['hy_upper'],
    })
    print(forecast_df.to_string(index=False))

    # Future forecast plot with MC Dropout CIs
    ctx     = daily_ts.tail(90)
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(ctx['date'], ctx['price'],
            color='#555555', lw=1.5, label='Historical (last 90 days)')
    ax.plot(forecast_df['date'], forecast_df['arima'],
            color=PALETTE['arima'], ls='--', lw=1.5, label='ARIMA')
    ax.plot(forecast_df['date'], forecast_df['standalone_lstm'],
            color=PALETTE['lstm'],  ls='-.', lw=1.5, label='Standalone LSTM')
    ax.fill_between(forecast_df['date'],
                    forecast_df['standalone_lower95'],
                    forecast_df['standalone_upper95'],
                    color=PALETTE['lstm'], alpha=0.08,
                    label='SA-LSTM 95% CI (MC Dropout)')
    ax.plot(forecast_df['date'], forecast_df['hybrid'],
            color=PALETTE['hybrid'], lw=2, label='Hybrid')
    ax.fill_between(forecast_df['date'],
                    forecast_df['hybrid_lower95'],
                    forecast_df['hybrid_upper95'],
                    color=PALETTE['hybrid'], alpha=0.10,
                    label='Hybrid 95% CI (MC Dropout)')
    ax.axvline(last_date, color='gray', ls=':', lw=2)
    ax.set_title(f'{FORECAST_HORIZON}-Day Ahead Price Forecast — All Models\n'
                 '(Shaded bands = 95% MC Dropout confidence intervals)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date'); ax.set_ylabel('Price per Pax (USD)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('future_forecast_all_models.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── 16. Summary box ──────────────────────────────────────────────────────
    best_model = min(all_metrics,
                     key=lambda m: all_metrics[m]['MAE'])
    best_mae   = all_metrics[best_model]['MAE']
    naive_mae  = all_metrics['Naïve (Seasonal)']['MAE']
    improv_vs_naive = (naive_mae - best_mae) / naive_mae * 100

    W = 62
    print("\n" + "╔" + "═" * W + "╗")
    print("║" + "  FINAL RESULTS SUMMARY".center(W) + "║")
    print("╠" + "═" * W + "╣")
    print(f"║  ARIMA order      : {arima_order}".ljust(W + 1) + "║")
    print(f"║  LSTM seq length  : {SEQUENCE_LENGTH} days".ljust(W + 1) + "║")
    print(f"║  Forecast horizon : {FORECAST_HORIZON} days".ljust(W + 1) + "║")
    print("╠" + "═" * W + "╣")
    print("║  TEST SET PERFORMANCE".ljust(W + 1) + "║")
    for name, m in all_metrics.items():
        line = (f"║  {name:22s}  "
                f"MAE={m['MAE']:7.3f}  RMSE={m['RMSE']:7.3f}  "
                f"MAPE={m['MAPE']:5.2f}%")
        print(line.ljust(W + 1) + "║")
    print("╠" + "═" * W + "╣")
    print(f"║  Best model       : {best_model}".ljust(W + 1) + "║")
    print(f"║  MAE improvement vs Naïve: {improv_vs_naive:+.2f}%".ljust(
          W + 1) + "║")
    print("╚" + "═" * W + "╝")

    # ── 17. Save artifacts ────────────────────────────────────────────────────
    save_all_artifacts(
        sa_mdl, hy_mdl,
        arima_res_a['arima_full'],
        sa_data['price_scaler'],
        hy_data['res_scaler'],
        hy_data['exog_scaler'],
        encoders, arima_order, arima_seasonal_order, exog_cols,
        forecast_df)

    return {
        'standalone_lstm_model': sa_mdl,
        'hybrid_lstm_model'    : hy_mdl,
        'arima_model'          : arima_res_a['arima_full'],
        'all_metrics'          : all_metrics,
        'perf_table'           : perf_table,
        'dm_results'           : dm_results,
        'cv_arima'             : cv_arima,
        'cv_sa_lstm'           : cv_sa_lstm,
        'forecast_df'          : forecast_df,
        'arima_order'          : arima_order,
        'arima_seasonal_order' : arima_seasonal_order,
        'exog_cols'            : exog_cols,
        'shap_df'              : shap_df,
    }

# ============================================================================
# 20. ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    # ── Google Colab: mount Drive if running in Colab ─────────────────────
    try:
        import google.colab
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted ✓")
    except ImportError:
        # Running locally (VS Code, terminal, Jupyter) — no Drive mount needed
        print("Running locally — skipping Google Drive mount.")

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\nLoading data from: {DATA_PATH}")
    df_raw  = pd.read_excel(DATA_PATH)
    print(f"Loaded {len(df_raw):,} rows ✓")

    # ── Run full pipeline ──────────────────────────────────────────────────
    results = run_pipeline(df_raw)