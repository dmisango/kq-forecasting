#!/usr/bin/env python3
"""
================================================================================
AUTOMATED DATA PIPELINE — Kenya Airways Air Ticket Price Forecasting
================================================================================

PURPOSE
───────
Performs standardised preprocessing and feature engineering on incoming booking
data, runs all three trained models (ARIMA, Standalone LSTM, ARIMA-LSTM Hybrid)
to produce updated price forecasts, and persists results to a centralised
SQLite database for consumption by the Streamlit dashboard.

USAGE
─────
    python pipeline.py --data bookings.xlsx --artifact-dir artifacts

OUTPUTS
───────
    forecasting.db              — SQLite database with all results tables
    logs/pipeline_YYYYMMDD.log  — Execution log for audit trail

FIXES APPLIED (v2)
──────────────────
    1. fillna(method=...) replaced with ffill()/bfill() — pandas 2.x compatible
    2. Keras model loading wrapped in safe multi-format loader (.keras then .h5)
    3. Hybrid model input access made robust — handles single and dual input models
    4. Unused hashlib import removed
    5. write_actuals guarded against NaT departure dates
    6. aggregate_daily uses correct column reference after rename
    7. Logger uses getLogger properly to avoid duplicate handlers
================================================================================
"""

import os
import sys
import argparse
import logging
import sqlite3
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              mean_absolute_percentage_error, r2_score)
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / 'artifacts'
DB_PATH      = BASE_DIR / 'forecasting.db'
LOG_DIR      = BASE_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH  = 30
FORECAST_HORIZON = 30
SEASONAL_M       = 7
TARGET_COL       = 'price_per_pax'
DATE_COL         = 'Date of issue day'   # ← UPDATE if your column name differs
DEP_COL          = 'Departure date'      # ← UPDATE if your column name differs
ROUTE_COL        = 'Route'              # ← UPDATE if your column name differs

BOOKING_WINDOW_BANDS = {
    'Last Minute (0-7d)'      : (0,   7),
    'Short Advance (8-14d)'   : (8,   14),
    'Medium Advance (15-30d)' : (15,  30),
    'Long Advance (31-60d)'   : (31,  60),
    'Very Long (60d+)'        : (61,  9999),
}

SEASONS = {
    'High Season (Jan-Feb, Jul-Aug, Nov-Dec)': [1, 2, 7, 8, 11, 12],
    'Shoulder Season'                         : [3, 6, 9, 10],
    'Low Season (Apr-May)'                    : [4, 5],
}


# ═════════════════════════════════════════════════════════════════════════════
# LOGGING
# FIX 7: Use a named logger with explicit handler management to avoid
# duplicate log lines when basicConfig is called multiple times.
# ═════════════════════════════════════════════════════════════════════════════
def setup_logger(run_id: str) -> logging.Logger:
    log_path = LOG_DIR / f'pipeline_{run_id}.log'
    logger = logging.getLogger('kq_pipeline')
    logger.setLevel(logging.INFO)

    # Only add handlers if none exist yet (prevents duplicates on re-runs)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s  %(levelname)-8s  %(message)s')

        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    else:
        # Update file handler path for new run
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
                h.close()
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s  %(levelname)-8s  %(message)s'))
        logger.addHandler(fh)

    return logger


# ═════════════════════════════════════════════════════════════════════════════
# DATABASE LAYER
# ═════════════════════════════════════════════════════════════════════════════
def init_database(db_path: Path) -> sqlite3.Connection:
    """Initialise the SQLite database and create tables if they do not exist."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.executescript("""
    CREATE TABLE IF NOT EXISTS forecasts (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id          TEXT    NOT NULL,
        generated_at    TEXT    NOT NULL,
        route           TEXT    NOT NULL,
        forecast_date   TEXT    NOT NULL,
        model           TEXT    NOT NULL,
        predicted_price REAL    NOT NULL,
        lower_ci_95     REAL,
        upper_ci_95     REAL,
        horizon_day     INTEGER NOT NULL,
        UNIQUE(run_id, route, forecast_date, model)
    );

    CREATE TABLE IF NOT EXISTS performance_metrics (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id       TEXT    NOT NULL,
        computed_at  TEXT    NOT NULL,
        route        TEXT    NOT NULL,
        segment      TEXT    NOT NULL,
        segment_type TEXT    NOT NULL,
        model        TEXT    NOT NULL,
        mae          REAL,
        rmse         REAL,
        mape         REAL,
        r2           REAL,
        n_obs        INTEGER,
        UNIQUE(run_id, route, segment, model)
    );

    CREATE TABLE IF NOT EXISTS actuals (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        recorded_at     TEXT    NOT NULL,
        route           TEXT    NOT NULL,
        departure_date  TEXT    NOT NULL,
        actual_price    REAL    NOT NULL,
        booking_window  INTEGER,
        departure_month INTEGER,
        UNIQUE(route, departure_date)
    );

    CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_id           TEXT PRIMARY KEY,
        started_at       TEXT NOT NULL,
        finished_at      TEXT,
        status           TEXT NOT NULL DEFAULT 'RUNNING',
        data_path        TEXT,
        n_rows_raw       INTEGER,
        n_rows_processed INTEGER,
        n_routes         INTEGER,
        error_message    TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_forecasts_route_date
        ON forecasts (route, forecast_date);
    CREATE INDEX IF NOT EXISTS idx_perf_run_route
        ON performance_metrics (run_id, route, segment_type);
    CREATE INDEX IF NOT EXISTS idx_actuals_route_date
        ON actuals (route, departure_date);
    """)
    conn.commit()
    return conn


def log_run_start(conn, run_id, data_path):
    conn.execute(
        "INSERT OR REPLACE INTO pipeline_runs "
        "(run_id, started_at, status, data_path) VALUES (?,?,?,?)",
        (run_id, datetime.utcnow().isoformat(), 'RUNNING', str(data_path))
    )
    conn.commit()


def log_run_finish(conn, run_id, n_raw, n_proc, n_routes,
                   status='SUCCESS', error=None):
    conn.execute(
        "UPDATE pipeline_runs SET finished_at=?, status=?, "
        "n_rows_raw=?, n_rows_processed=?, n_routes=?, error_message=? "
        "WHERE run_id=?",
        (datetime.utcnow().isoformat(), status,
         n_raw, n_proc, n_routes, error, run_id)
    )
    conn.commit()


# ═════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════
def load_and_preprocess(data_path: str, log: logging.Logger):
    """Load and preprocess raw booking Excel data."""
    log.info(f"Loading data from {data_path}")
    df = pd.read_excel(data_path)
    n_raw = len(df)
    log.info(f"Raw rows: {n_raw:,}")

    # ── Print actual column names to help diagnose mismatches ─────────────────
    log.info(f"Columns found in file: {list(df.columns)}")

    # ── Date columns ──────────────────────────────────────────────────────────
    if DATE_COL not in df.columns:
        raise KeyError(
            f"Column '{DATE_COL}' not found. "
            f"Available columns: {list(df.columns)}\n"
            f"Edit DATE_COL at the top of pipeline.py to match your file."
        )
    if DEP_COL not in df.columns:
        raise KeyError(
            f"Column '{DEP_COL}' not found. "
            f"Available columns: {list(df.columns)}\n"
            f"Edit DEP_COL at the top of pipeline.py to match your file."
        )

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    df[DEP_COL]  = pd.to_datetime(df[DEP_COL],  errors='coerce')

    # ── Basic cleaning ────────────────────────────────────────────────────────
    df = df.drop_duplicates()
    df = df.dropna(subset=[DATE_COL, DEP_COL])

    # Handle revenue and pax columns flexibly
    revenue_col = 'Flown CPV'
    pax_col     = 'Flown seg pax'

    if revenue_col not in df.columns:
        # Try common alternatives
        for alt in ['Revenue', 'CPV', 'Fare', 'Total Revenue']:
            if alt in df.columns:
                revenue_col = alt
                log.info(f"Using '{alt}' as revenue column")
                break
        else:
            raise KeyError(
                f"Revenue column '{revenue_col}' not found. "
                f"Available: {list(df.columns)}"
            )

    if pax_col not in df.columns:
        for alt in ['Pax', 'Passengers', 'Seg Pax', 'PAX']:
            if alt in df.columns:
                pax_col = alt
                log.info(f"Using '{alt}' as pax column")
                break
        else:
            raise KeyError(
                f"Pax column '{pax_col}' not found. "
                f"Available: {list(df.columns)}"
            )

    df = df.dropna(subset=[revenue_col, pax_col])
    df = df[pd.to_numeric(df[pax_col],     errors='coerce') > 0]
    df = df[pd.to_numeric(df[revenue_col], errors='coerce') >= 0]
    df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce')
    df[pax_col]     = pd.to_numeric(df[pax_col],     errors='coerce')

    df['booking_window'] = (df[DEP_COL] - df[DATE_COL]).dt.days
    df = df[df['booking_window'] >= 0]

    df[TARGET_COL] = df[revenue_col] / df[pax_col]

    # ── Route column ──────────────────────────────────────────────────────────
    if ROUTE_COL not in df.columns:
        if 'Origin' in df.columns and 'Destination' in df.columns:
            df[ROUTE_COL] = df['Origin'].astype(str) + '-' + df['Destination'].astype(str)
            log.info("Route column created from Origin + Destination")
        else:
            df[ROUTE_COL] = 'ALL'
            log.info("No route column found — all data treated as single route 'ALL'")

    # ── Temporal features ─────────────────────────────────────────────────────
    df['booking_month']         = df[DATE_COL].dt.month
    df['booking_day_of_week']   = df[DATE_COL].dt.dayofweek
    df['booking_quarter']       = df[DATE_COL].dt.quarter
    df['booking_year']          = df[DATE_COL].dt.year
    df['is_weekend_booking']    = df['booking_day_of_week'].isin([5, 6]).astype(int)
    df['departure_month']       = df[DEP_COL].dt.month
    df['departure_day_of_week'] = df[DEP_COL].dt.dayofweek
    df['departure_year']        = df[DEP_COL].dt.year
    df['is_weekend_departure']  = df['departure_day_of_week'].isin([5, 6]).astype(int)
    df['is_high_season']        = df['departure_month'].isin(
                                      {1, 2, 7, 8, 11, 12}).astype(int)

    # ── Booking category ──────────────────────────────────────────────────────
    def _cat(d):
        if d <= 7:   return 'last_minute'
        if d <= 14:  return 'short_advance'
        if d <= 30:  return 'medium_advance'
        if d <= 60:  return 'long_advance'
        return 'very_long_advance'
    df['booking_category'] = df['booking_window'].apply(_cat)

    # ── Cyclical encoding ─────────────────────────────────────────────────────
    for col, period in [
        ('booking_month', 12), ('booking_day_of_week', 7),
        ('departure_month', 12), ('departure_day_of_week', 7)
    ]:
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)

    df['booking_window_log'] = np.log1p(df['booking_window'].clip(lower=0))
    df['days_since_start']   = (df[DATE_COL] - df[DATE_COL].min()).dt.days

    log.info(f"After preprocessing: {len(df):,} rows  ({n_raw - len(df):,} removed)")
    return df, n_raw


# ═════════════════════════════════════════════════════════════════════════════
# ARTIFACT LOADING
# FIX 2: Safe model loader — tries .keras format first, falls back to .h5,
# then to weights-only. This handles Keras version mismatches gracefully.
# ═════════════════════════════════════════════════════════════════════════════
def safe_load_model(path_stem: Path, log: logging.Logger):
    """
    Try loading a Keras model from multiple file formats in order of preference.
    path_stem: path WITHOUT extension, e.g. artifacts/standalone_lstm_model
    """
    candidates = [
        path_stem.parent / (path_stem.name + '.keras'),
        path_stem.parent / (path_stem.name + '.h5'),
        path_stem,  # exact path as-is
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                model = tf.keras.models.load_model(str(candidate), compile=False)
                log.info(f"Loaded model from {candidate.name}")
                return model
            except Exception as e:
                log.warning(f"Failed to load {candidate.name}: {e}")
                continue

    raise FileNotFoundError(
        f"Could not load model '{path_stem.name}' in any format (.keras or .h5). "
        f"Please re-save the model from Colab and place in the artifacts folder."
    )


def load_artifacts(artifact_dir: Path, log: logging.Logger) -> dict:
    """Load all trained model artifacts from disk."""
    log.info(f"Loading artifacts from {artifact_dir}")
    arts = {}

    # ── Scikit-learn / joblib artifacts ───────────────────────────────────────
    for key, filename in [
        ('arima',    'arima_model.pkl'),
        ('price_sc', 'price_scaler.pkl'),
        ('res_sc',   'residual_scaler.pkl'),
        ('exog_sc',  'exog_scaler.pkl'),
        ('encoders', 'label_encoders.pkl'),
        ('meta',     'model_meta.pkl'),
    ]:
        fpath = artifact_dir / filename
        if not fpath.exists():
            raise FileNotFoundError(
                f"Missing artifact: {fpath}\n"
                f"Make sure you have run the training script and copied all "
                f"artifact files into the '{artifact_dir}' folder."
            )
        arts[key] = joblib.load(fpath)

    # ── Keras models (FIX 2: safe multi-format loader) ────────────────────────
    arts['sa_lstm'] = safe_load_model(
        artifact_dir / 'standalone_lstm_model', log)
    arts['hy_lstm'] = safe_load_model(
        artifact_dir / 'hybrid_lstm_model', log)

    log.info("All artifacts loaded ✓")
    return arts


# ═════════════════════════════════════════════════════════════════════════════
# DAILY AGGREGATION (per route)
# FIX 1: Replace deprecated fillna(method=...) with ffill()/bfill()
# FIX 6: Use correct column name 'date' after groupby rename
# ═════════════════════════════════════════════════════════════════════════════
def aggregate_daily(df: pd.DataFrame, route: str) -> pd.DataFrame:
    """Aggregate bookings to daily median price for a single route."""
    sub = df[df[ROUTE_COL] == route] if route != 'ALL' else df.copy()

    if sub.empty:
        return pd.DataFrame(columns=['date', 'price'])

    # Group by departure date → daily median price
    agg = (sub.groupby(DEP_COL)[TARGET_COL]
              .median()
              .reset_index())
    agg.columns = ['date', 'price']   # rename here — 'date' used consistently below
    agg = agg.sort_values('date').reset_index(drop=True)

    if len(agg) == 0:
        return pd.DataFrame(columns=['date', 'price'])

    # Fill gaps in date range (missing days get forward-filled price)
    full_range = pd.date_range(agg['date'].min(), agg['date'].max(), freq='D')
    agg = (agg.set_index('date')
              .reindex(full_range))
    agg.index.name = 'date'

    # FIX 1: Use ffill()/bfill() instead of deprecated fillna(method=...)
    agg['price'] = agg['price'].ffill().bfill()

    return agg.reset_index()


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE — ARIMA
# ═════════════════════════════════════════════════════════════════════════════
def run_arima_forecast(arima_model, prices: np.ndarray,
                       horizon: int, log: logging.Logger) -> tuple:
    """Refit ARIMA on latest prices then forecast horizon steps ahead."""
    try:
        order = (arima_model.order
                 if hasattr(arima_model, 'order')
                 else (1, 1, 1))
        fitted = ARIMA(prices.astype(float), order=order).fit()
        fc     = fitted.forecast(steps=horizon)
        resid_std = float(np.std(fitted.resid))
        log.debug(f"ARIMA{order} refitted on {len(prices)} points")
        return np.array(fc), resid_std, fitted
    except Exception as e:
        log.warning(f"ARIMA failed ({e}) — using naive seasonal baseline")
        idx = min(SEASONAL_M, len(prices))
        fc  = np.array([prices[-idx]] * horizon)
        return fc, float(np.std(prices)), None


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE — STANDALONE LSTM
# ═════════════════════════════════════════════════════════════════════════════
def run_sa_lstm_forecast(sa_model, price_sc,
                          prices: np.ndarray,
                          horizon: int,
                          seq_len: int) -> np.ndarray:
    """Recursive autoregressive forecast using standalone LSTM."""
    scaled = price_sc.transform(prices.reshape(-1, 1)).flatten()
    hist   = list(scaled[-seq_len:])
    preds  = []

    for _ in range(horizon):
        win = np.array(hist[-seq_len:]).reshape(1, seq_len, 1)
        p_s = float(sa_model.predict(win, verbose=0)[0, 0])
        preds.append(float(price_sc.inverse_transform([[p_s]])[0, 0]))
        hist.append(p_s)

    return np.array(preds)


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE — HYBRID
# FIX 3: Robust input detection — handles both single-input and dual-input
# hybrid models without crashing on hy_model.input[1]
# ═════════════════════════════════════════════════════════════════════════════
def run_hybrid_forecast(arima_fc: np.ndarray,
                        arima_model,
                        hy_model,
                        res_sc,
                        prices: np.ndarray,
                        horizon: int,
                        seq_len: int) -> np.ndarray:
    """ARIMA + LSTM residual correction hybrid forecast."""
    if arima_model is None:
        return arima_fc

    try:
        resid = np.array(arima_model.resid)
        res_s = res_sc.transform(resid.reshape(-1, 1)).flatten()
        hist  = list(res_s[-seq_len:])

        # FIX 3: Detect number of model inputs safely
        try:
            model_inputs = hy_model.input
            if isinstance(model_inputs, list):
                n_exog = model_inputs[1].shape[-1]
                dual_input = True
            else:
                dual_input = False
                n_exog = 0
        except Exception:
            dual_input = False
            n_exog = 0

        exog_ph = np.zeros((1, seq_len, n_exog)) if dual_input else None

        corr = []
        for _ in range(horizon):
            rw  = np.array(hist[-seq_len:]).reshape(1, seq_len, 1)
            if dual_input and exog_ph is not None:
                c_s = float(hy_model.predict([rw, exog_ph], verbose=0)[0, 0])
            else:
                c_s = float(hy_model.predict(rw, verbose=0)[0, 0])
            corr.append(float(res_sc.inverse_transform([[c_s]])[0, 0]))
            hist.append(c_s)

        return arima_fc + np.array(corr)

    except Exception as e:
        # If hybrid correction fails, return plain ARIMA forecast
        return arima_fc


# ═════════════════════════════════════════════════════════════════════════════
# CONFIDENCE INTERVALS
# ═════════════════════════════════════════════════════════════════════════════
def empirical_ci(predictions: np.ndarray, resid_std: float,
                 z: float = 1.96) -> tuple:
    """95% CI using empirical residual standard deviation."""
    lower = predictions - z * resid_std
    upper = predictions + z * resid_std
    return lower, upper


# ═════════════════════════════════════════════════════════════════════════════
# PERFORMANCE COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE/RMSE/MAPE/R² dict; None values on insufficient data."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    n = min(len(y_true), len(y_pred))
    if n < 2:
        return {'mae': None, 'rmse': None, 'mape': None, 'r2': None, 'n': n}
    yt, yp = y_true[:n], y_pred[:n]
    return {
        'mae' : float(mean_absolute_error(yt, yp)),
        'rmse': float(np.sqrt(mean_squared_error(yt, yp))),
        'mape': float(mean_absolute_percentage_error(yt, yp) * 100),
        'r2'  : float(r2_score(yt, yp)),
        'n'   : n
    }


def evaluate_disaggregated(df_actuals: pd.DataFrame,
                            df_preds: pd.DataFrame,
                            route: str) -> list:
    """Metrics disaggregated by booking window and season."""
    rows = []
    now  = datetime.utcnow().isoformat()

    merged = df_actuals.merge(
        df_preds[['departure_date', 'model', 'predicted_price']],
        on='departure_date', how='inner')

    if merged.empty:
        return rows

    for model in merged['model'].unique():
        sub = merged[merged['model'] == model]

        # Overall
        m = compute_metrics(sub['actual_price'].values,
                            sub['predicted_price'].values)
        rows.append({**m, 'route': route, 'segment': 'Overall',
                     'segment_type': 'overall', 'model': model,
                     'computed_at': now})

        # By booking window
        if 'booking_window' in sub.columns:
            for band, (lo, hi) in BOOKING_WINDOW_BANDS.items():
                mask = sub['booking_window'].between(lo, hi)
                if mask.sum() > 1:
                    m = compute_metrics(
                        sub.loc[mask, 'actual_price'].values,
                        sub.loc[mask, 'predicted_price'].values)
                    rows.append({**m, 'route': route, 'segment': band,
                                 'segment_type': 'booking_window',
                                 'model': model, 'computed_at': now})

        # By season
        if 'departure_month' in sub.columns:
            for season, months in SEASONS.items():
                mask = sub['departure_month'].isin(months)
                if mask.sum() > 1:
                    m = compute_metrics(
                        sub.loc[mask, 'actual_price'].values,
                        sub.loc[mask, 'predicted_price'].values)
                    rows.append({**m, 'route': route, 'segment': season,
                                 'segment_type': 'season',
                                 'model': model, 'computed_at': now})

    return rows


# ═════════════════════════════════════════════════════════════════════════════
# DATABASE WRITE HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def write_forecasts(conn, run_id, route, forecast_dates, fc_dict, resid_std):
    """Write forecast rows for all three models to the database."""
    now  = datetime.utcnow().isoformat()
    rows = []
    for model_name, preds in fc_dict.items():
        lower, upper = empirical_ci(preds, resid_std)
        for i, (dt, p, lo, hi) in enumerate(
                zip(forecast_dates, preds, lower, upper), 1):
            rows.append((run_id, now, route,
                          dt.strftime('%Y-%m-%d'),
                          model_name,
                          float(p), float(lo), float(hi), i))
    conn.executemany(
        "INSERT OR REPLACE INTO forecasts "
        "(run_id, generated_at, route, forecast_date, model, "
        "predicted_price, lower_ci_95, upper_ci_95, horizon_day) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        rows)
    conn.commit()


def write_actuals(conn, df: pd.DataFrame, route: str):
    """
    Upsert observed prices into the actuals table.
    FIX 5: Skip rows where departure date is NaT or price is invalid.
    """
    now  = datetime.utcnow().isoformat()
    rows = []
    for _, row in df.iterrows():
        dep_date = row.get(DEP_COL)
        price    = row.get(TARGET_COL)

        # Guard against NaT dates and invalid prices
        if pd.isna(dep_date) or pd.isna(price):
            continue
        try:
            date_str = pd.Timestamp(dep_date).strftime('%Y-%m-%d')
            rows.append((now, route, date_str, float(price),
                          int(row.get('booking_window', 0)),
                          int(row.get('departure_month', 0))))
        except Exception:
            continue

    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO actuals "
            "(recorded_at, route, departure_date, actual_price, "
            "booking_window, departure_month) VALUES (?,?,?,?,?,?)",
            rows)
        conn.commit()


def write_performance(conn, run_id, rows):
    """Write disaggregated performance metrics to the database."""
    conn.executemany(
        "INSERT OR REPLACE INTO performance_metrics "
        "(run_id, computed_at, route, segment, segment_type, model, "
        "mae, rmse, mape, r2, n_obs) "
        "VALUES (:run_id, :computed_at, :route, :segment, :segment_type, "
        ":model, :mae, :rmse, :mape, :r2, :n)",
        [{**r, 'run_id': run_id} for r in rows])
    conn.commit()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def run_pipeline(data_path: str,
                 artifact_dir: Path = ARTIFACT_DIR,
                 db_path: Path = DB_PATH,
                 routes: list = None) -> None:
    """
    Full pipeline:
      1. Preprocess raw booking data
      2. Load trained model artifacts
      3. For each route: forecast, write to DB, compute performance metrics
      4. Log completion
    """
    run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    log    = setup_logger(run_id)

    log.info('=' * 60)
    log.info(f"Pipeline run : {run_id}")
    log.info(f"Data path   : {data_path}")
    log.info(f"DB path     : {db_path}")
    log.info('=' * 60)

    conn   = init_database(db_path)
    log_run_start(conn, run_id, data_path)

    n_raw = n_proc = n_routes = 0

    try:
        # 1. Preprocess
        df, n_raw = load_and_preprocess(data_path, log)
        n_proc    = len(df)

        # 2. Load artifacts
        arts    = load_artifacts(artifact_dir, log)
        seq_len = arts['meta'].get('seq_len', SEQUENCE_LENGTH)

        # 3. Determine routes to process
        if routes is None:
            route_list = sorted(df[ROUTE_COL].dropna().unique().tolist())
            routes     = ['ALL'] + route_list
        log.info(f"Routes to process: {len(routes)}  — {routes[:8]}"
                 f"{'...' if len(routes) > 8 else ''}")

        # 4. Route loop
        for route in routes:
            log.info(f"  Processing route: {route}")
            daily = aggregate_daily(df, route)

            if len(daily) < seq_len + FORECAST_HORIZON:
                log.warning(
                    f"  Skipping {route}: only {len(daily)} daily points "
                    f"(need {seq_len + FORECAST_HORIZON})")
                continue

            prices         = daily['price'].values
            last_date      = daily['date'].max()
            forecast_dates = pd.date_range(
                last_date + timedelta(days=1),
                periods=FORECAST_HORIZON, freq='D')

            # Run all three models
            arima_fc, resid_std, arima_fitted = run_arima_forecast(
                arts['arima'], prices, FORECAST_HORIZON, log)

            sa_fc = run_sa_lstm_forecast(
                arts['sa_lstm'], arts['price_sc'],
                prices, FORECAST_HORIZON, seq_len)

            hy_fc = run_hybrid_forecast(
                arima_fc, arima_fitted,
                arts['hy_lstm'], arts['res_sc'],
                prices, FORECAST_HORIZON, seq_len)

            # Write to database
            write_forecasts(conn, run_id, route, forecast_dates,
                            {'ARIMA'            : arima_fc,
                             'Standalone LSTM'  : sa_fc,
                             'ARIMA-LSTM Hybrid': hy_fc},
                            resid_std)

            # Write actuals
            route_df = df[df[ROUTE_COL] == route] if route != 'ALL' else df
            write_actuals(conn, route_df.tail(500), route)

            # Disaggregated performance metrics
            actuals_db = pd.read_sql_query(
                "SELECT departure_date, actual_price, booking_window, "
                "departure_month FROM actuals WHERE route=?",
                conn, params=(route,))

            if not actuals_db.empty:
                preds_db = pd.read_sql_query(
                    "SELECT forecast_date AS departure_date, model, "
                    "predicted_price FROM forecasts "
                    "WHERE run_id=? AND route=?",
                    conn, params=(run_id, route))

                # Enrich actuals with booking metadata
                meta_cols = [DEP_COL, 'booking_window', 'departure_month']
                meta_df   = (route_df[meta_cols]
                             .drop_duplicates(DEP_COL)
                             .copy())
                meta_df[DEP_COL] = pd.to_datetime(
                    meta_df[DEP_COL]).dt.strftime('%Y-%m-%d')
                meta_df = meta_df.rename(columns={DEP_COL: 'departure_date'})

                actuals_db = actuals_db.merge(meta_df,
                                              on='departure_date',
                                              how='left',
                                              suffixes=('', '_meta'))
                # Use meta columns where original is missing
                for col in ['booking_window', 'departure_month']:
                    if f'{col}_meta' in actuals_db.columns:
                        actuals_db[col] = actuals_db[col].fillna(
                            actuals_db[f'{col}_meta'])
                        actuals_db.drop(columns=[f'{col}_meta'], inplace=True)

                perf_rows = evaluate_disaggregated(actuals_db, preds_db, route)
                if perf_rows:
                    write_performance(conn, run_id, perf_rows)

            n_routes += 1
            log.info(f"  ✓ {route}: forecasts written to database")

        log_run_finish(conn, run_id, n_raw, n_proc, n_routes)
        log.info('=' * 60)
        log.info(f"Pipeline COMPLETE — run_id={run_id}  routes={n_routes}")
        log.info(f"Database: {db_path}")
        log.info('=' * 60)

    except Exception as e:
        log.error(f"Pipeline FAILED: {e}", exc_info=True)
        log_run_finish(conn, run_id, n_raw, n_proc, n_routes,
                       status='FAILED', error=str(e))
        raise
    finally:
        conn.close()


# ═════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kenya Airways Air Ticket Price Forecasting Pipeline')
    parser.add_argument('--data',         required=True,
                        help='Path to raw booking Excel file')
    parser.add_argument('--artifact-dir', default=str(ARTIFACT_DIR),
                        help='Directory containing trained model artifacts')
    parser.add_argument('--db',           default=str(DB_PATH),
                        help='Path to SQLite database file')
    parser.add_argument('--routes',       nargs='*', default=None,
                        help='Subset of routes to process (default: all)')
    args = parser.parse_args()

    run_pipeline(
        data_path    = args.data,
        artifact_dir = Path(args.artifact_dir),
        db_path      = Path(args.db),
        routes       = args.routes,
    )
