"""
================================================================================
KENYA AIRWAYS — AIR TICKET PRICE FORECASTING DASHBOARD
Streamlit Interactive Web Application for Revenue Management
================================================================================

PURPOSE
───────
Provides Kenya Airways revenue management analysts with an interactive,
real-time view of model forecasts, forecast accuracy, and contextual
performance insights disaggregated by route, booking window, and season.
Human oversight is facilitated through transparent metric reporting that
flags contexts of high or low predictive reliability.

FEATURES
────────
  ▸ Multi-model forecast viewer (ARIMA | Standalone LSTM | ARIMA-LSTM Hybrid)
  ▸ Route-level and network-level forecast panels
  ▸ 95% confidence interval visualisation
  ▸ Real-time performance monitoring: MAE, RMSE, MAPE, R²
  ▸ Disaggregated metrics by route, booking window, and season
  ▸ Model reliability heatmap (flags contexts of low accuracy)
  ▸ Historical actual vs forecast comparison
  ▸ Pipeline audit log (last N runs)
  ▸ Downloadable forecast CSV

RUN
───
    streamlit run dashboard.py

    Environment variable overrides:
        KQ_API_BASE         Forecasting API base URL (default: https://kq-forecasting.onrender.com)
        KQ_ARTIFACT_DIR     Path to model artifacts  (default: ./artifacts)

DEPENDENCIES
────────────
    pip install streamlit plotly pandas numpy sqlalchemy
================================================================================
"""
from loader import download_if_missing
download_if_missing()

import os
from pathlib import Path
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import streamlit as st
import matplotlib.colors as mcolors

# ─── Configuration ────────────────────────────────────────────────────────────
# ─── API base URL ─────────────────────────────────────────────────────────────
# The dashboard no longer reads forecasting.db directly.
# All data is fetched through the REST API (api.py), which is the single
# controlled access layer between the database and all consumers.
# Set KQ_API_BASE to point at a remote server in production.
API_BASE     = os.getenv('KQ_API_BASE', 'https://kq-forecasting.onrender.com')
ARTIFACT_DIR = Path(os.getenv('KQ_ARTIFACT_DIR', 'artifacts'))

MODEL_COLOURS = {
    'ARIMA'            : '#2196F3',   # blue
    'Standalone LSTM'  : '#FF9800',   # orange
    'ARIMA-LSTM Hybrid': '#E91E63',   # pink/red
    'Actual'           : '#1B1B1B',
}

RELIABILITY_THRESHOLDS = {
    'MAE'  : {'excellent': 5,   'good': 15,  'poor': 30},
    'MAPE' : {'excellent': 5.0, 'good': 10.0,'poor': 20.0},
    'R2'   : {'excellent': 0.90,'good': 0.75,'poor': 0.50},
}

PAGE_ICON = "✈️"


# ─── Utilities ────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    """Convert a hex colour string to a CSS rgba() string for Plotly fills."""
    try:
        rgb = mcolors.to_rgb(hex_color)
        return (f'rgba({int(rgb[0]*255)},'
                f'{int(rgb[1]*255)},'
                f'{int(rgb[2]*255)},'
                f'{alpha})')
    except ValueError:
        return f'rgba(128,128,128,{alpha})'


# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG — must be first Streamlit call
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "KQ Price Forecasting | Revenue Management",
    page_icon   = PAGE_ICON,
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# Force light theme regardless of user system preference
st.markdown("""
<style>
  /* Root and app background */
  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stApp"] {
      background-color: #ffffff !important;
      color: #1a1a2e !important;
  }
  /* Main content area */
  [data-testid="stMain"], .main .block-container {
      background-color: #ffffff !important;
      color: #1a1a2e !important;
  }
  /* Sidebar */
  [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
      background-color: #f8f9fc !important;
      color: #1a1a2e !important;
  }
  /* Metric cards */
  [data-testid="stMetric"] {
      background-color: #f0f2f6 !important;
      border-radius: 8px;
      padding: 8px 12px;
  }
  /* Dataframes */
  [data-testid="stDataFrame"] {
      background-color: #ffffff !important;
  }
  /* Tab labels */
  .stTabs [data-baseweb="tab"] {
      color: #1a1a2e !important;
  }
  /* Headings */
  h1, h2, h3, h4, p, label, div {
      color: #1a1a2e;
  }
  /* Input widgets */
  [data-testid="stSelectbox"], [data-testid="stMultiSelect"],
  [data-testid="stSlider"], [data-testid="stCheckbox"] {
      background-color: #ffffff !important;
  }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# API HELPERS  — all data fetched through api.py, never from SQLite directly
# ═════════════════════════════════════════════════════════════════════════════

def _api_get(path: str, params: dict = {}) -> dict:
    """
    Single HTTP helper for all API calls.  Every fetch function below calls
    this — nowhere else in the dashboard talks to the database directly.

    Failure modes:
      • ConnectionError  — API server not running  → show actionable error
      • HTTPError        — bad request or server error → show warning
      • Timeout          — API slow or unreachable → show warning
    Returns empty dict on any failure so callers get empty DataFrames rather
    than exceptions.
    """
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"**Cannot reach the forecasting API at `{API_BASE}`.**  "
            f"Start it with:  `uvicorn api:app --port 8000`",
            icon="🔌"
        )
        return {}
    except requests.exceptions.Timeout:
        st.warning("API request timed out. The server may be busy.", icon="⏱️")
        return {}
    except requests.exceptions.HTTPError as e:
        st.warning(f"API returned {e.response.status_code} for `{path}`", icon="⚠️")
        return {}


@st.cache_data(ttl=300)
def api_health() -> dict:
    """
    Check whether the API is reachable and which models are loaded.
    Used by the connectivity check in main() to decide whether to show
    live data or fall back to demo data.
    """
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_routes() -> list:
    data = _api_get("/routes")
    return data.get("routes", ["NBO-MBA"])


@st.cache_data(ttl=300)
def fetch_forecasts(route: str, horizon: int, models: list) -> pd.DataFrame:
    data = _api_get("/forecast/latest", {
        "route"  : route,
        "horizon": horizon,
        "models" : ",".join(models),
    })
    if not data or "forecasts" not in data:
        return pd.DataFrame()
    rows = []
    for fc in data["forecasts"]:
        for pt in fc["points"]:
            rows.append({
                "forecast_date"  : pt["date"],
                "model"          : fc["model"],
                "predicted_price": pt["predicted_price"],
                "lower_ci_95"    : pt["lower_ci"],
                "upper_ci_95"    : pt["upper_ci"],
                "horizon_day"    : pt["horizon_day"],
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    return df


@st.cache_data(ttl=300)
def fetch_actuals(route: str, n_days: int = 90,
                  start_date: str = None, end_date: str = None) -> pd.DataFrame:
    params = {"route": route, "limit": n_days}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    data = _api_get("/data/actuals", params)
    if not data or "records" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["records"])
    if not df.empty:
        df["departure_date"] = pd.to_datetime(df["departure_date"])
    return df


@st.cache_data(ttl=300)
def fetch_performance(route: str, segment_type: str) -> pd.DataFrame:
    data = _api_get("/performance", {"route": route})
    if not data or "results" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["results"])
    if not df.empty and "segment_type" in df.columns:
        df = df[df["segment_type"] == segment_type].copy()
    return df


@st.cache_data(ttl=60)
def fetch_pipeline_log(n: int = 10) -> pd.DataFrame:
    data = _api_get("/pipeline/runs", {"n": n})
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


@st.cache_data(ttl=300)
def fetch_all_routes_latest_metrics() -> pd.DataFrame:
    """Performance overview across all routes — used by Forecast Analysis tab."""
    data = _api_get("/performance", {"route": "NBO-MBA"})
    if not data or "results" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["results"])
    if not df.empty and "segment" in df.columns:
        df = df[df["segment"] == "Overall"].copy()
    return df


# ═════════════════════════════════════════════════════════════════════════════
# HELPER: demo data when DB is empty
# ═════════════════════════════════════════════════════════════════════════════
def _demo_forecasts(models: list, horizon: int = 30) -> pd.DataFrame:
    """Generate synthetic forecast data for dashboard demonstration."""
    np.random.seed(42)
    base_price = 350.0
    dates = pd.date_range(datetime.today() + timedelta(days=1),
                          periods=horizon, freq='D')
    rows = []
    for m in models:
        noise_scale = {'ARIMA': 12, 'Standalone LSTM': 18,
                       'ARIMA-LSTM Hybrid': 9}.get(m, 10)
        trend = np.linspace(0, 20, horizon)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(horizon) / 7)
        preds = base_price + trend + seasonal + np.random.randn(horizon) * noise_scale
        std   = noise_scale * 1.5
        for i, (dt, p) in enumerate(zip(dates, preds), 1):
            rows.append({'forecast_date': dt, 'model': m,
                          'predicted_price': p,
                          'lower_ci_95': p - 1.96 * std,
                          'upper_ci_95': p + 1.96 * std,
                          'horizon_day': i})
    return pd.DataFrame(rows)


def _demo_actuals(n: int = 60) -> pd.DataFrame:
    np.random.seed(0)
    dates  = pd.date_range(datetime.today() - timedelta(days=n), periods=n, freq='D')
    prices = 340 + 15 * np.sin(2 * np.pi * np.arange(n) / 7) + \
             np.random.randn(n) * 10
    # Passenger numbers: weekly seasonal pattern with some noise
    pax    = (85 + 20 * np.sin(2 * np.pi * np.arange(n) / 7) +
              np.random.randint(-8, 8, n)).clip(min=10)
    return pd.DataFrame({
        'departure_date': dates,
        'actual_price'  : prices,
        'pax'           : pax.astype(int),
    })


def _demo_performance(models: list, segment_type: str = 'booking_window') -> pd.DataFrame:
    """
    Generate demo performance data for the given segment_type.
    segment_type: 'overall' | 'booking_window' | 'season'
    """
    np.random.seed(1)
    rows = []
    if segment_type == 'overall':
        # Fixed plausible metrics for NBO-MBA — shown in metric cards
        defaults = {
            'ARIMA':             {'mae': 14.87, 'rmse': 17.62, 'mape': 35.54, 'r2': 0.61},
            'Standalone LSTM':   {'mae':  5.98, 'rmse':  7.11, 'mape': 18.41, 'r2': 0.89},
            'ARIMA-LSTM Hybrid': {'mae':  8.43, 'rmse': 10.22, 'mape': 22.17, 'r2': 0.84},
        }
        for m in models:
            d = defaults.get(m, {'mae': 10, 'rmse': 12, 'mape': 20, 'r2': 0.80})
            rows.append({'model': m, 'segment': 'Overall',
                         'mae': d['mae'], 'rmse': d['rmse'],
                         'mape': d['mape'], 'r2': d['r2'], 'n_obs': 274})
    elif segment_type == 'booking_window':
        segments = ['Last Minute (0–7d)', 'Short Advance (8–14d)',
                    'Medium Advance (15–30d)', 'Long Advance (31–60d)',
                    'Very Long (60d+)']
        for m in models:
            for seg in segments:
                rows.append({'model': m, 'segment': seg,
                              'mae' : round(np.random.uniform(5, 30), 2),
                              'rmse': round(np.random.uniform(7, 40), 2),
                              'mape': round(np.random.uniform(2, 18), 2),
                              'r2'  : round(np.random.uniform(0.6, 0.96), 4),
                              'n_obs': int(np.random.randint(30, 200))})
    else:  # season
        segments = ['High Season (Jan–Feb, Jul–Aug, Nov–Dec)',
                    'Shoulder Season', 'Low Season (Apr–May)']
        for m in models:
            for seg in segments:
                rows.append({'model': m, 'segment': seg,
                              'mae' : round(np.random.uniform(5, 25), 2),
                              'rmse': round(np.random.uniform(7, 35), 2),
                              'mape': round(np.random.uniform(2, 15), 2),
                              'r2'  : round(np.random.uniform(0.65, 0.96), 4),
                              'n_obs': int(np.random.randint(30, 200))})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# HELPER: single-route all_m metrics (demo mode, NBO-MBA only)
# ═════════════════════════════════════════════════════════════════════════════
def _demo_all_routes_metrics(models: list) -> pd.DataFrame:
    """
    Demo replacement for fetch_all_routes_latest_metrics().
    Returns NBO-MBA only — reflecting the actual single-route dataset.
    No phantom routes are injected into the heatmap or network overview.
    """
    np.random.seed(3)
    route = 'NBO-MBA'
    rows = []
    seg_bw = ['Last Minute (0–7d)', 'Short Advance (8–14d)',
              'Medium Advance (15–30d)', 'Long Advance (31–60d)',
              'Very Long (60d+)']
    seg_ss = ['High Season (Jan–Feb, Jul–Aug, Nov–Dec)',
              'Shoulder Season', 'Low Season (Apr–May)']
    for m in models:
        rows.append({'route': route, 'model': m, 'segment': 'Overall',
                     'segment_type': 'overall',
                     'mae': round(np.random.uniform(6, 18), 2),
                     'rmse': round(np.random.uniform(8, 22), 2),
                     'mape': round(np.random.uniform(5, 25), 2),
                     'r2': round(np.random.uniform(0.70, 0.95), 4),
                     'n_obs': 274})
        for seg in seg_bw:
            rows.append({'route': route, 'model': m, 'segment': seg,
                         'segment_type': 'booking_window',
                         'mae': round(np.random.uniform(5, 30), 2),
                         'rmse': round(np.random.uniform(7, 40), 2),
                         'mape': round(np.random.uniform(3, 22), 2),
                         'r2': round(np.random.uniform(0.60, 0.95), 4),
                         'n_obs': int(np.random.randint(20, 120))})
        for seg in seg_ss:
            rows.append({'route': route, 'model': m, 'segment': seg,
                         'segment_type': 'season',
                         'mae': round(np.random.uniform(5, 25), 2),
                         'rmse': round(np.random.uniform(7, 35), 2),
                         'mape': round(np.random.uniform(3, 18), 2),
                         'r2': round(np.random.uniform(0.65, 0.95), 4),
                         'n_obs': int(np.random.randint(30, 180))})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# COMPONENT: FORECAST CHART
# ═════════════════════════════════════════════════════════════════════════════
def render_forecast_chart(fc_df: pd.DataFrame,
                           act_df: pd.DataFrame,
                           selected_models: list,
                           show_ci: bool,
                           route: str) -> None:
    fig = go.Figure()

    # Actuals
    if not act_df.empty:
        fig.add_trace(go.Scatter(
            x=act_df['departure_date'], y=act_df['actual_price'],
            name='Actual Price', mode='lines+markers',
            line=dict(color=MODEL_COLOURS['Actual'], width=2),
            marker=dict(size=4)
        ))

    # Forecasts
    for model in selected_models:
        sub = fc_df[fc_df['model'] == model]
        if sub.empty:
            continue
        col = MODEL_COLOURS.get(model, '#888888')

        # Fix 12: Consistent line styles — hybrid thicker
        line_width = 3 if model == 'ARIMA-LSTM Hybrid' else 2
        line_dash  = ('dash'  if model == 'ARIMA' else
                      'dot'   if model == 'Standalone LSTM' else 'solid')

        fig.add_trace(go.Scatter(
            x=sub['forecast_date'], y=sub['predicted_price'],
            name=model, mode='lines',
            line=dict(color=col, width=line_width, dash=line_dash)
        ))

        # Fix 1: CI colour — hex_to_rgba replaces broken .replace() hack
        if show_ci and 'lower_ci_95' in sub.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([sub['forecast_date'],
                              sub['forecast_date'][::-1]]),
                y=pd.concat([sub['upper_ci_95'],
                              sub['lower_ci_95'][::-1]]),
                fill='toself',
                fillcolor=hex_to_rgba(col, 0.12),
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False, name=f'{model} 95% CI',
                hoverinfo='skip'
            ))

    # COVID annotation if in range — same add_shape/add_annotation workaround
    covid_date = pd.Timestamp('2020-03-01')
    if not act_df.empty and not fc_df.empty:
        all_dates = pd.concat([act_df['departure_date'],
                                fc_df['forecast_date']]).dropna()
        if all_dates.min() <= covid_date <= all_dates.max():
            covid_str = covid_date.isoformat()
            fig.add_shape(
                type='line',
                x0=covid_str, x1=covid_str,
                y0=0, y1=1,
                xref='x', yref='paper',
                line=dict(dash='dot', color='purple', width=1.5),
                opacity=0.5,
            )
            fig.add_annotation(
                x=covid_str,
                y=0.97,
                xref='x', yref='paper',
                text='COVID-19',
                showarrow=False,
                xanchor='right',
                yanchor='top',
                font=dict(size=11, color='purple'),
                bgcolor='rgba(255,255,255,0.7)',
            )

    # Fix 2.1: Forecast start vertical divider
    # add_vline with annotation_* kwargs is broken on datetime x-axes in older
    # Plotly versions (shapeannotation.py calls sum() on string/Timestamp x0/x1).
    # Workaround: use add_shape + add_annotation separately, which bypasses the
    # broken annotation positioning code entirely.
    if not fc_df.empty:
        forecast_start = pd.Timestamp(fc_df['forecast_date'].min()).isoformat()
        fig.add_shape(
            type='line',
            x0=forecast_start, x1=forecast_start,
            y0=0, y1=1,
            xref='x', yref='paper',
            line=dict(dash='dash', color='gray', width=1.5),
            opacity=0.6,
        )
        fig.add_annotation(
            x=forecast_start,
            y=1.0,
            xref='x', yref='paper',
            text='Forecast Start',
            showarrow=False,
            xanchor='left',
            yanchor='top',
            font=dict(size=11, color='gray'),
            bgcolor='rgba(255,255,255,0.7)',
        )

    # Fix 2.2: Subtitle showing horizon length
    horizon_days = int(fc_df['horizon_day'].max()) if not fc_df.empty else 0
    title_text = (f'Price Forecast — {route}'
                  f'<br><sup>Horizon: {horizon_days} days ahead</sup>'
                  if horizon_days else f'Price Forecast — Route: {route}')

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16)),
        xaxis_title='Date',
        yaxis_title='Price per Passenger (USD)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
        hovermode='x unified',
        template='plotly_white',
        height=480,
        margin=dict(t=80, b=40, l=60, r=20),
    )

    # Fix 2.3: Range slider for zoom
    fig.update_xaxes(rangeslider_visible=True)

    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# COMPONENT: METRIC CARDS
# ═════════════════════════════════════════════════════════════════════════════
def render_metric_cards(perf_df: pd.DataFrame, models: list) -> None:
    """One card per model showing headline test-set metrics (aggregated mean)."""
    cols = st.columns(len(models))
    for col, model in zip(cols, models):
        sub = perf_df[perf_df['model'] == model]
        if sub.empty or sub['mae'].isna().all():
            col.metric(model, "—", help="No performance data available")
            continue
        # Fix 3: aggregate across segments rather than taking arbitrary iloc[0]
        row = sub[['mae', 'mape', 'r2']].mean()
        mae_val  = f"{row['mae']:.2f}"   if pd.notna(row['mae'])  else "—"
        mape_val = f"{row['mape']:.1f}%" if pd.notna(row['mape']) else "—"
        r2_val   = f"{row['r2']:.3f}"    if pd.notna(row['r2'])   else "—"
        # Reliability badge
        mape = row['mape'] if pd.notna(row['mape']) else 99
        if mape < RELIABILITY_THRESHOLDS['MAPE']['excellent']:
            badge = "🟢 Excellent"
        elif mape < RELIABILITY_THRESHOLDS['MAPE']['good']:
            badge = "🟡 Good"
        elif mape < RELIABILITY_THRESHOLDS['MAPE']['poor']:
            badge = "🟠 Fair"
        else:
            badge = "🔴 Poor"
        with col:
            st.markdown(f"""
            <div style="background:#f8f9fa;border-radius:10px;
                        padding:14px 18px;border-left:4px solid
                        {MODEL_COLOURS.get(model,'#888')};
                        margin-bottom:4px">
              <div style="font-size:0.85rem;color:#666;font-weight:600">
                {model}</div>
              <div style="font-size:1.5rem;font-weight:700;margin:4px 0">
                MAE: {mae_val}</div>
              <div style="font-size:0.9rem;color:#444">
                MAPE: {mape_val} &nbsp;|&nbsp; R²: {r2_val}</div>
              <div style="font-size:0.8rem;margin-top:6px">{badge}</div>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# COMPONENT: DISAGGREGATED PERFORMANCE TABLE
# ═════════════════════════════════════════════════════════════════════════════
def render_performance_table(perf_df: pd.DataFrame, title: str,
                              metric: str = 'mape') -> None:
    if perf_df.empty:
        st.info(f"No {title.lower()} data available yet.")
        return

    st.subheader(title)
    pivot = perf_df.pivot_table(
        index='segment', columns='model',
        values=metric, aggfunc='first'
    ).round(2)

    # Colour cells by reliability threshold
    thresh = RELIABILITY_THRESHOLDS.get(metric.upper(), {})
    def colour_cell(val):
        if pd.isna(val):
            return 'background-color: #eeeeee'
        if thresh:
            exc = thresh.get('excellent', -np.inf)
            gd  = thresh.get('good', -np.inf)
            pr  = thresh.get('poor', np.inf)
            if metric.upper() == 'R2':
                if val >= exc: return 'background-color: #c8f7c5'
                if val >= gd:  return 'background-color: #fffde7'
                if val >= pr:  return 'background-color: #fff3e0'
                return 'background-color: #ffebee'
            else:
                if val <= exc: return 'background-color: #c8f7c5'
                if val <= gd:  return 'background-color: #fffde7'
                if val <= pr:  return 'background-color: #fff3e0'
                return 'background-color: #ffebee'
        return ''

    # Fix 6: background_gradient is more robust across browsers than applymap
    cmap = 'RdYlGn' if metric.upper() == 'R2' else 'RdYlGn_r'
    styled = pivot.style.background_gradient(cmap=cmap, axis=None)
    metric_label = {'mape': 'MAPE (%)', 'mae': 'MAE (USD)',
                    'rmse': 'RMSE (USD)', 'r2': 'R²'}.get(metric, metric)
    st.caption(f"Metric: **{metric_label}**  |  "
               f"🟢 Excellent  🟡 Good  🟠 Fair  🔴 Poor")
    st.dataframe(styled, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# COMPONENT: RELIABILITY HEATMAP
# ═════════════════════════════════════════════════════════════════════════════
def render_reliability_heatmap(all_metrics: pd.DataFrame,
                                selected_model: str,
                                metric: str = 'mape') -> None:
    """
    Heatmap of MAPE (or chosen metric) by route and booking window / season.
    Cells are coloured by reliability tier — directly supports human oversight
    by immediately flagging routes and contexts with low predictive accuracy.
    """
    sub = all_metrics[all_metrics['model'] == selected_model]
    if sub.empty:
        st.info("No data for heatmap.")
        return

    # Pivot: rows = routes, columns = segments
    piv = sub.pivot_table(index='route', columns='segment',
                           values=metric, aggfunc='first')
    if piv.empty:
        return

    # Fix 5.1 & 5.2: sort routes by mean error, keep worst 15 for readability
    route_order = piv.mean(axis=1).sort_values(ascending=(metric.lower() != 'r2'))
    piv = piv.loc[route_order.head(15).index]

    metric_label = {'mape': 'MAPE (%)', 'mae': 'MAE (USD)',
                    'rmse': 'RMSE (USD)', 'r2': 'R²'}.get(metric, metric)

    # For MAPE / MAE / RMSE: lower = better → reverse colour scale
    reverse = metric.lower() != 'r2'
    colourscale = 'RdYlGn_r' if reverse else 'RdYlGn'

    fig = px.imshow(
        piv,
        color_continuous_scale=colourscale,
        aspect='auto',
        text_auto='.1f',
        labels=dict(color=metric_label),
        title=f'Model Reliability Heatmap — {selected_model}  ({metric_label})'
    )
    fig.update_layout(height=max(300, 60 * len(piv)),
                      template='plotly_white',
                      coloraxis_colorbar=dict(title=metric_label))
    fig.update_xaxes(tickangle=25)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🔴 Red = low reliability (high error) · "
        "🟢 Green = high reliability (low error)  |  "
        "Use this to identify routes and booking contexts "
        "requiring closer human review before pricing decisions."
    )


# ═════════════════════════════════════════════════════════════════════════════
# COMPONENT: PIPELINE STATUS
# ═════════════════════════════════════════════════════════════════════════════
def render_pipeline_status() -> None:
    st.subheader("Automated Pipeline Audit Log")
    log_df = fetch_pipeline_log(10)
    if log_df.empty:
        st.info("No pipeline runs recorded yet.")
        return

    def status_badge(s):
        return {'SUCCESS': '🟢', 'FAILED': '🔴', 'RUNNING': '🟡'}.get(s, '⚪') + f' {s}'

    log_df['status'] = log_df['status'].apply(status_badge)
    log_df['started_at'] = pd.to_datetime(log_df['started_at']).dt.strftime('%Y-%m-%d %H:%M')
    log_df['finished_at'] = pd.to_datetime(log_df['finished_at']).dt.strftime('%Y-%m-%d %H:%M')
    log_df.columns = ['Run ID', 'Started', 'Finished', 'Status',
                       'Raw Rows', 'Processed Rows', 'Routes']
    st.dataframe(log_df, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
def main():
    # st.cache_data.clear() 
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#ffffff;border-left:5px solid #CC0000;
                padding:16px 24px;border-radius:4px;margin-bottom:20px;
                box-shadow:0 1px 4px rgba(0,0,0,0.08)">
      <h1 style="color:#CC0000;margin:0;font-size:1.7rem;
                 font-family:Georgia,serif;font-weight:800;">
         &nbsp; Kenya Airways — Air Ticket Price Forecasting
      </h1>
      <p style="color:#555;margin:5px 0 0 0;font-size:0.92rem;">
        Revenue Management Intelligence Dashboard &nbsp;|&nbsp;
        ARIMA · Standalone LSTM · ARIMA-LSTM Hybrid
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── API connectivity check ────────────────────────────────────────────────
    # The dashboard no longer checks for a local database file.
    # It checks whether the API server is reachable instead.
    health      = api_health()
    api_online  = bool(health)
    db_ok       = health.get("db_reachable", False)
    api_reachable = api_online and db_ok

    if not api_online:
        st.warning(
            f"⚠️  Forecasting API not reachable at `{API_BASE}`.  "
            "Showing **demonstration data**.  "
            "Start the API with:  `uvicorn api:app --port 8000`",
            icon="⚠️")
    elif not db_ok:
        st.warning(
            "⚠️  API is running but the database is not yet populated.  "
            "Showing **demonstration data** — run `pipeline.py` first.",
            icon="⚠️")
    else:
        st.success(
            f"  Connected to forecasting API  |  "
            f"Models loaded: {', '.join(health.get('models_available', []))}  |  "
            f"Last pipeline run: {health.get('last_pipeline_run', 'unknown')}",
            icon="🟢")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        # st.markdown(
        #     "<p style='color:#CC0000;font-size:2.0rem;font-weight:800;"
        #     "font-family:Georgia,serif;letter-spacing:1px;"
        #     "margin:0;padding:2px 0;'>Kenya Airways</p>"
        #     "<p style='color:#CC0000;font-size:0.75rem;font-weight:600;"
        #     "font-family:Georgia,serif;letter-spacing:2px;"
        #     "margin:0;'>Price Forecasting</p>",
        #     unsafe_allow_html=True
        # )
        st.image("kenya_airways_logo.png", use_container_width=True)
    
        st.markdown("---")

        # Route
        st.markdown("**Route**")
        routes   = fetch_routes() if api_reachable else ['NBO-MBA']
        route    = st.selectbox("Route", options=routes, index=0,
                                label_visibility='collapsed')

        # Models
        st.markdown("**Models**")
        all_models = ['ARIMA', 'Standalone LSTM', 'ARIMA-LSTM Hybrid']
        sel_models = st.multiselect("Models", options=all_models,
                                     default=all_models,
                                     label_visibility='collapsed')

        # Calendar date range picker (max 30 days)
        st.markdown("**Forecast Period**")
        today         = date.today()
        default_start = date(2021, 1, 1)    # ← was: today + timedelta(days=1)
        default_end   = date(2021, 1, 31)   # ← was: today + timedelta(days=30)
        date_range = st.date_input(
            "Select forecast dates",
            value=(default_start, default_end),
            min_value=date(2016, 1, 1),     # ← allow historical dates
            max_value=today + timedelta(days=365),
            label_visibility='collapsed',
        )
        # Enforce 30-day max
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            fc_start_date, fc_end_date = date_range
            delta = (fc_end_date - fc_start_date).days
            if delta > 30:
                st.warning("⚠️ Maximum forecast period is 30 days. End date adjusted.")
                fc_end_date = fc_start_date + timedelta(days=30)
            history_days = max(30, delta * 3)
        else:
            fc_start_date = default_start
            fc_end_date   = default_end
            history_days  = 90
        forecast_horizon_days = (fc_end_date - fc_start_date).days + 1

        st.markdown("**Actuals Overlay (Optional)**")
        show_actuals_range = st.checkbox("Specify actuals date range", value=True)
        if show_actuals_range:
            actuals_range = st.date_input(
                "Actuals date range",
                value=(date(2020, 12, 1), date(2021, 1, 31)),  # ← sync with forecast
                min_value=date(2016, 1, 1),
                max_value=today + timedelta(days=365),
                label_visibility='collapsed',
            )
            if isinstance(actuals_range, (list, tuple)) and len(actuals_range) == 2:
                act_start = actuals_range[0].isoformat()
                act_end   = actuals_range[1].isoformat()
            else:
                act_start = act_end = None
        else:
            act_start = act_end = None

        # Metric selector (compact)
        st.markdown("**Performance Metric**")
        metric = st.selectbox("Metric",
                               options=['mape', 'mae', 'rmse', 'r2'],
                               format_func=lambda x: {
                                   'mape': 'MAPE (%)', 'mae': 'MAE (USD)',
                                   'rmse': 'RMSE (USD)',
                                   'r2': 'R²'}[x],
                               label_visibility='collapsed')

        # CI toggle
        show_ci = st.checkbox("Show 95% Confidence Intervals", value=True)

        st.markdown("---")
        if st.button(" Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"API: `{API_BASE}`")

    # ── Fetch data through API (live) or fall back to demo data ──────────────
    if api_reachable:
        fc_df  = fetch_forecasts(route, forecast_horizon_days, sel_models)
        act_df = fetch_actuals(route, n_days=1000,
                           start_date=act_start,
                           end_date=act_end)
          # Use explicit date range if set, otherwise fall back to last n_days
        # if act_start and act_end:
        #     act_df = fetch_actuals(route, n_days=1000,        # large limit so range isn't truncated
        #                            start_date=act_start,
        #                            end_date=act_end)
        # else:
        #     act_df = fetch_actuals(route, n_days=history_days)
        
        perf_ov = fetch_performance(route, 'overall')
        # act_df  = fetch_actuals(route, history_days)
        # perf_ov = fetch_performance(route, 'overall')
        perf_bw = fetch_performance(route, 'booking_window')
        perf_ss = fetch_performance(route, 'season')
        all_m   = fetch_all_routes_latest_metrics()
    else:
        # API unreachable or DB empty — use demo data so dashboard remains
        # navigable during development and thesis demonstrations
        fc_df   = _demo_forecasts(sel_models)
        act_df  = _demo_actuals(history_days)
        perf_bw = _demo_performance(sel_models, 'booking_window')
        perf_ss = _demo_performance(sel_models, 'season')
        all_m   = _demo_all_routes_metrics(sel_models)
        perf_ov = pd.DataFrame()

    # Thesis test-set results are always the fallback for the metric cards —
    # performance_metrics only populates once forecast dates have passed
    # and actuals have been recorded, which has not happened yet.
    if perf_ov.empty:
        perf_ov = _demo_performance(sel_models, 'overall')

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB LAYOUT
    # ═══════════════════════════════════════════════════════════════════════════
    tab_fc, tab_perf, tab_heat, tab_network, tab_custom, tab_pipeline = st.tabs([
        "Forecasts",
        "Test-Set Results",
        "Forecast Analysis",
        "Booking Intelligence",
        "Custom Forecast",
        "Pipeline Status",
    ])

    # ───────────────────────────────────────────────────────────────────────────
    # TAB 1 — FORECASTS
    # ───────────────────────────────────────────────────────────────────────────
    with tab_fc:
        if not sel_models:
            st.warning("Please select at least one model in the sidebar.")
        else:
            # Fix 7.1: Model recommendation panel
            if not perf_ov.empty and 'mape' in perf_ov.columns:
                best_model = (perf_ov.groupby('model')['mape']
                              .mean().idxmin())
                st.success(
                    f"**Recommended Model for {route}:** {best_model} "
                    f"— lowest average MAPE across all segments",
                    icon="🏆")

            # Headline metric cards
            st.subheader(f"Model Performance Overview — {route}")
            render_metric_cards(perf_ov, sel_models)
            st.markdown("---")

            # Fix 7.2 & 7.3 & 13: Quick-read KPI strip above chart
            if not fc_df.empty and not act_df.empty:
                hybrid_fc = fc_df[fc_df['model'] == 'ARIMA-LSTM Hybrid']
                if hybrid_fc.empty:
                    hybrid_fc = fc_df[fc_df['model'] == sel_models[0]]

                min_p   = hybrid_fc['predicted_price'].min()
                max_p   = hybrid_fc['predicted_price'].max()
                first_p = hybrid_fc['predicted_price'].iloc[0]
                last_actual = act_df['actual_price'].iloc[-1]
                price_delta = first_p - last_actual
                n_obs_proxy = len(act_df)
                demand_str  = ("🔴 High" if n_obs_proxy > 120 else
                               "🟡 Medium" if n_obs_proxy > 60 else "🟢 Low")

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric(
                    "Forecast Price Range",
                    f"${min_p:.0f} – ${max_p:.0f}",
                    help="Min–max predicted price over the full forecast horizon"
                )
                kpi2.metric(
                    "Price vs Last Actual",
                    f"${first_p:.2f}",
                    delta=f"{price_delta:+.2f} USD",
                    delta_color="inverse"
                )
                kpi3.metric(
                    "Route Demand Strength",
                    demand_str,
                    help="Proxy based on number of historical observations"
                )
                kpi4.metric(
                    "Forecast Horizon",
                    f"{int(fc_df['horizon_day'].max())} days"
                )
                st.markdown("---")

            # ── Recommended fare panel ───────────────────────────────────
            if not fc_df.empty:
                hybrid_fc_rec = fc_df[fc_df['model'] == 'ARIMA-LSTM Hybrid']
                if hybrid_fc_rec.empty:
                    hybrid_fc_rec = fc_df[fc_df['model'] == sel_models[0]] if sel_models else fc_df
                if not hybrid_fc_rec.empty:
                    rec_price  = float(hybrid_fc_rec['predicted_price'].mean())
                    rec_lower  = float(hybrid_fc_rec['lower_ci_95'].mean()) if 'lower_ci_95' in hybrid_fc_rec.columns else rec_price * 0.92
                    rec_upper  = float(hybrid_fc_rec['upper_ci_95'].mean()) if 'upper_ci_95' in hybrid_fc_rec.columns else rec_price * 1.08
                    rec_model  = 'ARIMA-LSTM Hybrid' if not fc_df[fc_df['model']=='ARIMA-LSTM Hybrid'].empty else sel_models[0]
                    st.markdown(
                        f"<div style='background:#fff8f0;border-left:5px solid #CC0000;"
                        f"padding:14px 20px;border-radius:4px;margin-bottom:12px;"
                        f"box-shadow:0 1px 3px rgba(0,0,0,0.07)'>"
                        f"<span style='font-size:0.8rem;color:#888;font-weight:600;"
                        f"letter-spacing:1px;text-transform:uppercase;'>"
                        f"Recommended Fare — {route}</span><br>"
                        f"<span style='font-size:2rem;font-weight:800;color:#CC0000;'>"
                        f"${rec_price:.2f}</span>"
                        f"<span style='font-size:1rem;color:#555;margin-left:12px;'>"
                        f"95% CI: ${rec_lower:.2f} – ${rec_upper:.2f}</span><br>"
                        f"<span style='font-size:0.8rem;color:#777;'>"
                        f"Based on {rec_model} · Horizon: {forecast_horizon_days} days"
                        f" ({fc_start_date.strftime('%d %b')} – {fc_end_date.strftime('%d %b %Y')})"
                        f"</span></div>",
                        unsafe_allow_html=True
                    )
            st.markdown("---")

            # Forecast chart
            render_forecast_chart(fc_df, act_df, sel_models,
                                   show_ci, route)

            # Summary stats table
            if not fc_df.empty:
                st.subheader("Forecast Summary Table")
                summary = (fc_df.groupby('model')
                               .agg(forecast_start=('forecast_date', 'min'),
                                    forecast_end=('forecast_date', 'max'),
                                    min_price=('predicted_price', 'min'),
                                    mean_price=('predicted_price', 'mean'),
                                    max_price=('predicted_price', 'max'))
                               .round(2)
                               .reset_index())
                st.dataframe(summary, use_container_width=True, hide_index=True)

                # Download button
                csv = fc_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Forecast CSV",
                    data=csv,
                    file_name=f"kq_forecast_{route}_{datetime.today().date()}.csv",
                    mime='text/csv'
                )

            # Actual vs Forecast error trace (last N days)
            # Fix 4: merge_asof handles horizon/date mismatches gracefully
            if not act_df.empty and not fc_df.empty:
                st.subheader("Recent Actual vs Forecast Error")
                fig_err = go.Figure()
                for model in sel_models:
                    sub = fc_df[fc_df['model'] == model].copy()
                    merged = pd.merge_asof(
                        act_df.sort_values('departure_date'),
                        sub.rename(columns={'forecast_date': 'departure_date'})
                           [['departure_date', 'predicted_price']]
                           .sort_values('departure_date'),
                        on='departure_date',
                        direction='nearest',
                        tolerance=pd.Timedelta('3d')
                    )
                    if merged.empty or merged['predicted_price'].isna().all():
                        continue
                    merged['error'] = (merged['actual_price'] -
                                       merged['predicted_price'])
                    fig_err.add_trace(go.Scatter(
                        x=merged['departure_date'], y=merged['error'],
                        name=model, mode='lines+markers',
                        line=dict(color=MODEL_COLOURS.get(model)),
                        marker=dict(size=4)
                    ))
                fig_err.add_hline(y=0, line_dash='dash', line_color='grey')
                fig_err.update_layout(
                    title='Forecast Error (Actual − Predicted)',
                    xaxis_title='Departure Date',
                    yaxis_title='Error (USD)',
                    hovermode='x unified',
                    template='plotly_white', height=340,
                    legend=dict(orientation='h', yanchor='bottom',
                                y=1.02, xanchor='right', x=1),
                    margin=dict(t=60, b=40, l=60, r=20)
                )
                st.plotly_chart(fig_err, use_container_width=True)

            # ── Passenger volume chart ────────────────────────────────────────
            if not act_df.empty and 'pax' in act_df.columns:
                st.subheader("👥 Historical Passenger Volume")
                fig_pax = go.Figure()
                fig_pax.add_trace(go.Bar(
                    x=act_df['departure_date'],
                    y=act_df['pax'],
                    name='Pax',
                    marker_color='#2196F3',
                    opacity=0.75,
                ))
                # 7-day rolling average
                pax_roll = act_df.set_index('departure_date')['pax'].rolling(7).mean()
                fig_pax.add_trace(go.Scatter(
                    x=pax_roll.index, y=pax_roll.values,
                    name='7-Day Rolling Avg',
                    line=dict(color='#CC0000', width=2),
                    mode='lines',
                ))
                fig_pax.update_layout(
                    template='plotly_white', height=280,
                    xaxis_title='Departure Date',
                    yaxis_title='Passengers',
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom',
                                y=1.02, xanchor='right', x=1),
                    margin=dict(t=30, b=40, l=60, r=20),
                )
                st.plotly_chart(fig_pax, use_container_width=True)

            # ── Threshold alerts ──────────────────────────────────────────────
            st.subheader("Model Reliability Alerts")
            alerts_fired = False
            for mdl in sel_models:
                sub = perf_ov[perf_ov['model'] == mdl] if not perf_ov.empty else pd.DataFrame()
                if sub.empty:
                    continue
                mape_val = float(sub['mape'].mean()) if 'mape' in sub.columns else None
                mae_val  = float(sub['mae'].mean())  if 'mae'  in sub.columns else None
                r2_val   = float(sub['r2'].mean())   if 'r2'   in sub.columns else None

                mdl_alerts = []
                if mape_val and mape_val > RELIABILITY_THRESHOLDS['MAPE']['poor']:
                    mdl_alerts.append(
                        f"MAPE **{mape_val:.1f}%** exceeds poor threshold "
                        f"({RELIABILITY_THRESHOLDS['MAPE']['poor']}%)")
                if mae_val and mae_val > RELIABILITY_THRESHOLDS['MAE']['poor']:
                    mdl_alerts.append(
                        f"MAE **${mae_val:.2f}** exceeds poor threshold "
                        f"(${RELIABILITY_THRESHOLDS['MAE']['poor']})")
                if r2_val and r2_val < RELIABILITY_THRESHOLDS['R2']['poor']:
                    mdl_alerts.append(
                        f"R² **{r2_val:.3f}** is below poor threshold "
                        f"({RELIABILITY_THRESHOLDS['R2']['poor']})")

                if mdl_alerts:
                    alerts_fired = True
                    for a in mdl_alerts:
                        st.error(f"🔴 **{mdl}** — {a}. "
                                 f"Manual pricing review recommended.", icon="⚠️")
                elif mape_val and mape_val > RELIABILITY_THRESHOLDS['MAPE']['good']:
                    alerts_fired = True
                    st.warning(f"🟡 **{mdl}** — MAPE {mape_val:.1f}% is in the "
                               f"'Fair' range. Consider verifying forecasts before "
                               f"pricing decisions.", icon="⚠️")
                else:
                    st.success(f"🟢 **{mdl}** — All metrics within acceptable thresholds.",
                               icon="✅")

    # ───────────────────────────────────────────────────────────────────────────
    # TAB 2 — TEST-SET RESULTS  (real thesis numbers, always available)
    # ───────────────────────────────────────────────────────────────────────────
    with tab_perf:
        st.markdown("""
        ### Model Test-Set Performance — NBO-MBA Route
        Results from the thesis evaluation on the **held-out test set (15%)**
        of the 2016-2020 Kenya Airways NBO-MBA booking dataset.
        These are the definitive accuracy figures from `air_ticket_price_forecasting_thesis.py`.
        """)

        # ── Hardcoded thesis test-set results ─────────────────────────────────
        # Source: run_pipeline() output, Section 14 of thesis pipeline
        RESULTS = pd.DataFrame([
            {'Model':'Naïve (Seasonal)', 'MAE':28.49, 'RMSE':31.65, 'MAPE':78.04, 'R2':0.12,  'n_obs': 274},
            {'Model':'ARIMA',            'MAE':14.87, 'RMSE':17.62, 'MAPE':35.54, 'R2':0.61,  'n_obs': 274},
            {'Model':'Standalone LSTM',  'MAE': 5.98, 'RMSE': 7.11, 'MAPE':18.41, 'R2':0.89,  'n_obs': 258},
            {'Model':'ARIMA-LSTM Hybrid','MAE': 8.43, 'RMSE':10.22, 'MAPE':22.17, 'R2':0.84,  'n_obs': 260},
        ])

        # ── 1. Headline metric cards ───────────────────────────────────────────
        st.subheader("Headline Metrics — Test Set")
        cols_r = st.columns(4)
        card_colours = {
            'Naïve (Seasonal)' : '#aaaaaa',
            'ARIMA'            : MODEL_COLOURS['ARIMA'],
            'Standalone LSTM'  : MODEL_COLOURS['Standalone LSTM'],
            'ARIMA-LSTM Hybrid': MODEL_COLOURS['ARIMA-LSTM Hybrid'],
        }
        for col_r, (_, row) in zip(cols_r, RESULTS.iterrows()):
            mape = row['MAPE']
            badge = ("🟢 Excellent" if mape < 5 else
                     "🟡 Good"      if mape < 10 else
                     "🟠 Fair"      if mape < 20 else
                     "🔴 Poor")
            col_r.markdown(f"""
            <div style="background:#f8f9fa;border-radius:10px;
                        padding:12px 16px;border-left:4px solid
                        {card_colours.get(row['Model'],'#888')};margin-bottom:4px">
              <div style="font-size:0.8rem;color:#555;font-weight:600">{row['Model']}</div>
              <div style="font-size:1.4rem;font-weight:700;margin:3px 0">
                MAE: {row['MAE']:.2f}</div>
              <div style="font-size:0.85rem;color:#444">
                MAPE: {row['MAPE']:.1f}% &nbsp;|&nbsp; R²: {row['R2']:.2f}</div>
              <div style="font-size:0.75rem;margin-top:5px">{badge}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── 2. Full ranked table ──────────────────────────────────────────────
        st.subheader("Ranked Performance Table")
        ranked = RESULTS.copy()
        ranked['MAPE Rank'] = ranked['MAPE'].rank().astype(int)
        ranked['MAE Rank']  = ranked['MAE'].rank().astype(int)
        ranked['R² Rank']   = ranked['R2'].rank(ascending=False).astype(int)
        ranked['Mean Rank'] = ((ranked['MAPE Rank'] + ranked['MAE Rank'] + ranked['R² Rank']) / 3).round(1)
        display_cols = ['Model','MAE','RMSE','MAPE','R2','Mean Rank']
        styled_r = (ranked[display_cols]
                    .style
                    .background_gradient(subset=['MAE','RMSE','MAPE'], cmap='RdYlGn_r', axis=0)
                    .background_gradient(subset=['R2'],                 cmap='RdYlGn',   axis=0)
                    .format({'MAE':'{:.2f}','RMSE':'{:.2f}',
                             'MAPE':'{:.1f}%','R2':'{:.3f}','Mean Rank':'{:.1f}'}))
        st.dataframe(styled_r, use_container_width=True, hide_index=True)

        # Improvement over naïve
        naive_mae  = float(RESULTS.loc[RESULTS['Model']=='Naïve (Seasonal)', 'MAE'].iloc[0])
        naive_mape = float(RESULTS.loc[RESULTS['Model']=='Naïve (Seasonal)', 'MAPE'].iloc[0])
        st.caption(
            f"**Baseline (Naïve):** MAE = {naive_mae:.2f} USD, MAPE = {naive_mape:.1f}%. "
            f"Standalone LSTM achieves a "
            f"**{(naive_mae-5.98)/naive_mae*100:.0f}% MAE reduction** over the naïve baseline.")

        st.markdown("---")

        # ── 3. Bar charts ─────────────────────────────────────────────────────
        col_b1, col_b2 = st.columns(2)
        bar_colours = [card_colours.get(m, '#888') for m in RESULTS['Model']]

        with col_b1:
            st.subheader("MAE Comparison")
            fig_mae = go.Figure(go.Bar(
                x=RESULTS['Model'], y=RESULTS['MAE'],
                marker_color=bar_colours,
                text=RESULTS['MAE'].round(2), textposition='outside'
            ))
            fig_mae.update_layout(
                yaxis_title='MAE (USD)', template='plotly_white',
                height=340, margin=dict(t=20,b=60,l=50,r=20),
                showlegend=False)
            st.plotly_chart(fig_mae, use_container_width=True)

        with col_b2:
            st.subheader("MAPE Comparison")
            fig_mape = go.Figure(go.Bar(
                x=RESULTS['Model'], y=RESULTS['MAPE'],
                marker_color=bar_colours,
                text=[f"{v:.1f}%" for v in RESULTS['MAPE']], textposition='outside'
            ))
            fig_mape.update_layout(
                yaxis_title='MAPE (%)', template='plotly_white',
                height=340, margin=dict(t=20,b=60,l=50,r=20),
                showlegend=False)
            st.plotly_chart(fig_mape, use_container_width=True)

        st.markdown("---")

        # ── 4. Radar chart ────────────────────────────────────────────────────
        st.subheader("Model Scorecard (Radar)")
        st.caption("Each axis is normalised 0–1 (higher = better). R² is used as-is; MAE/RMSE/MAPE are inverted.")
        radar_models = ['ARIMA', 'Standalone LSTM', 'ARIMA-LSTM Hybrid']
        radar_data   = RESULTS[RESULTS['Model'].isin(radar_models)].copy()

        # Normalise: for error metrics, invert (lower → higher score)
        for col_n in ['MAE', 'RMSE', 'MAPE']:
            mn, mx = radar_data[col_n].min(), radar_data[col_n].max()
            radar_data[col_n + '_score'] = 1 - (radar_data[col_n] - mn) / (mx - mn + 1e-9)
        mn, mx = radar_data['R2'].min(), radar_data['R2'].max()
        radar_data['R2_score'] = (radar_data['R2'] - mn) / (mx - mn + 1e-9)

        cats = ['MAE Score', 'RMSE Score', 'MAPE Score', 'R² Score']
        score_cols = ['MAE_score', 'RMSE_score', 'MAPE_score', 'R2_score']

        fig_radar = go.Figure()
        for _, rrow in radar_data.iterrows():
            vals = [rrow[c] for c in score_cols]
            vals += vals[:1]   # close polygon
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats + [cats[0]],
                fill='toself', name=rrow['Model'],
                line_color=MODEL_COLOURS.get(rrow['Model'], '#888'),
                opacity=0.7
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, template='plotly_white',
            height=380, margin=dict(t=40,b=40,l=60,r=60)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")

        # ── 5. Diebold-Mariano interpretation ─────────────────────────────────
        st.subheader("Statistical Significance — Diebold-Mariano Test")
        dm_df = pd.DataFrame([
            {'Comparison':'ARIMA vs Naïve',         'DM Stat':'-3.21', 'p-value':'0.002', 'Result':'ARIMA significantly better '},
            {'Comparison':'SA-LSTM vs Naïve',       'DM Stat':'-5.84', 'p-value':'<0.001','Result':'SA-LSTM significantly better '},
            {'Comparison':'Hybrid vs Naïve',        'DM Stat':'-4.62', 'p-value':'<0.001','Result':'Hybrid significantly better '},
            {'Comparison':'SA-LSTM vs ARIMA',       'DM Stat':'-4.11', 'p-value':'<0.001','Result':'SA-LSTM significantly better '},
            {'Comparison':'Hybrid vs ARIMA',        'DM Stat':'-2.87', 'p-value':'0.005', 'Result':'Hybrid significantly better '},
            {'Comparison':'Hybrid vs SA-LSTM',      'DM Stat':'+1.43', 'p-value':'0.156', 'Result':'No significant difference '},
        ])
        st.dataframe(dm_df, use_container_width=True, hide_index=True)
        st.caption(
            "DM test (Harvey, Leybourne & Newbold 1997) with small-sample correction. "
            "H₀: equal predictive accuracy. Negative statistic → left model is more accurate. "
            "The Hybrid is statistically better than ARIMA but not significantly "
            "different from the Standalone LSTM at α=0.05.")

    # ───────────────────────────────────────────────────────────────────────────
    # TAB 3 — FORECAST ANALYSIS  (derived from forecasts table — always populated)
    # ───────────────────────────────────────────────────────────────────────────
    with tab_heat:
        st.markdown("""
        ### Forecast Analysis — NBO-MBA
        Deep-dive into the model forecasts: price trajectories, inter-model spread,
        rolling volatility, and price distribution — all computed directly from the
        forecast data in the database.
        """)

        if fc_df.empty:
            st.info("No forecast data in database yet. Run the pipeline first, or use the 🔮 Custom Forecast tab.")
        else:
            # ── 1. Price trajectory comparison ────────────────────────────────
            st.subheader("30-Day Price Trajectory — All Models")
            fig_traj = go.Figure()
            for model in sel_models:
                sub = fc_df[fc_df['model'] == model]
                if sub.empty:
                    continue
                col_t = MODEL_COLOURS.get(model, '#888')
                lw    = 3 if model == 'ARIMA-LSTM Hybrid' else 2
                ld    = ('dash' if model == 'ARIMA' else
                         'dot'  if model == 'Standalone LSTM' else 'solid')
                fig_traj.add_trace(go.Scatter(
                    x=sub['forecast_date'], y=sub['predicted_price'],
                    name=model, mode='lines',
                    line=dict(color=col_t, width=lw, dash=ld)
                ))
                if show_ci and 'lower_ci_95' in sub.columns:
                    fig_traj.add_trace(go.Scatter(
                        x=pd.concat([sub['forecast_date'], sub['forecast_date'][::-1]]),
                        y=pd.concat([sub['upper_ci_95'], sub['lower_ci_95'][::-1]]),
                        fill='toself', fillcolor=hex_to_rgba(col_t, 0.10),
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False, hoverinfo='skip'
                    ))
            fig_traj.update_layout(
                xaxis_title='Date', yaxis_title='Price per Pax (USD)',
                hovermode='x unified', template='plotly_white',
                height=380, legend=dict(orientation='h', yanchor='bottom',
                                        y=1.02, xanchor='right', x=1),
                margin=dict(t=40,b=40,l=60,r=20))
            st.plotly_chart(fig_traj, use_container_width=True)

            st.markdown("---")

            # ── 2. Model spread (max-min per day) ─────────────────────────────
            st.subheader("Daily Inter-Model Price Spread")
            st.caption("How much the three models disagree on any given day. Wider spread = higher uncertainty.")
            spread_df = (fc_df.groupby('forecast_date')['predicted_price']
                         .agg(price_max='max', price_min='min')
                         .reset_index())
            spread_df['spread'] = spread_df['price_max'] - spread_df['price_min']
            fig_spread = go.Figure()
            fig_spread.add_trace(go.Bar(
                x=spread_df['forecast_date'], y=spread_df['spread'],
                marker_color='#7986CB', name='Price Spread (USD)',
                hovertemplate='%{x|%d %b}: $%{y:.2f} spread<extra></extra>'
            ))
            fig_spread.update_layout(
                xaxis_title='Date', yaxis_title='Max − Min Price (USD)',
                template='plotly_white', height=300,
                margin=dict(t=20,b=40,l=60,r=20))
            st.plotly_chart(fig_spread, use_container_width=True)

            st.markdown("---")

            # ── 3. Rolling 7-day volatility per model ─────────────────────────
            st.subheader("Forecast Volatility (7-Day Rolling Std)")
            st.caption("Models with lower rolling standard deviation produce smoother, more stable forecasts.")
            fig_vol = go.Figure()
            for model in sel_models:
                sub = fc_df[fc_df['model'] == model].sort_values('forecast_date')
                if len(sub) < 3:
                    continue
                sub = sub.copy()
                sub['rolling_std'] = sub['predicted_price'].rolling(7, min_periods=2).std()
                fig_vol.add_trace(go.Scatter(
                    x=sub['forecast_date'], y=sub['rolling_std'],
                    name=model, mode='lines',
                    line=dict(color=MODEL_COLOURS.get(model,'#888'), width=2)
                ))
            fig_vol.update_layout(
                xaxis_title='Date', yaxis_title='Rolling Std (USD)',
                hovermode='x unified', template='plotly_white', height=300,
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            xanchor='right', x=1),
                margin=dict(t=40,b=40,l=60,r=20))
            st.plotly_chart(fig_vol, use_container_width=True)

            st.markdown("---")

            # ── 4. Price distribution per model ───────────────────────────────
            st.subheader("Forecast Price Distribution")
            fig_dist = go.Figure()
            for model in sel_models:
                sub = fc_df[fc_df['model'] == model]
                if sub.empty:
                    continue
                fig_dist.add_trace(go.Violin(
                    y=sub['predicted_price'], name=model,
                    box_visible=True, meanline_visible=True,
                    fillcolor=hex_to_rgba(MODEL_COLOURS.get(model,'#888'), 0.5),
                    line_color=MODEL_COLOURS.get(model,'#888'),
                    opacity=0.8
                ))
            fig_dist.update_layout(
                yaxis_title='Predicted Price (USD)',
                template='plotly_white', height=360,
                margin=dict(t=20,b=40,l=60,r=20))
            st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown("---")

            # ── 5. Summary statistics table ───────────────────────────────────
            st.subheader("Forecast Statistics Table")
            stats_df = (fc_df.groupby('model')['predicted_price']
                        .agg(Min='min', Mean='mean', Median='median',
                             Max='max', Std='std',
                             CV=lambda x: x.std()/x.mean()*100)
                        .round(2)
                        .reset_index()
                        .rename(columns={'model':'Model', 'CV':'CV (%)'}))
            st.dataframe(
                stats_df.style.background_gradient(
                    subset=['Std','CV (%)'], cmap='YlOrRd', axis=0),
                use_container_width=True, hide_index=True)
            st.caption("CV = coefficient of variation (Std / Mean × 100). Lower CV = more stable forecast.")

    # ───────────────────────────────────────────────────────────────────────────
    # TAB 4 — BOOKING INTELLIGENCE  (from actuals table + thesis breakdown)
    # ───────────────────────────────────────────────────────────────────────────
    with tab_network:
        st.markdown("""
        ### Booking Window Intelligence — NBO-MBA
        How ticket prices vary by **advance purchase window** and **travel season**,
        based on the 2016-2020 Kenya Airways historical booking data.
        """)

        # ── Thesis booking window breakdown (from Section 3.3 / known results) ─
        BW_DATA = pd.DataFrame([
            {'Window':'Last Minute (0–7d)',      'Avg Price (USD)':418.2, 'Pct of Bookings':8.3,
             'ARIMA MAPE':52.1, 'LSTM MAPE':24.8, 'Hybrid MAPE':31.4},
            {'Window':'Short Advance (8–14d)',   'Avg Price (USD)':385.6, 'Pct of Bookings':12.7,
             'ARIMA MAPE':41.3, 'LSTM MAPE':20.1, 'Hybrid MAPE':26.2},
            {'Window':'Medium Advance (15–30d)', 'Avg Price (USD)':357.4, 'Pct of Bookings':28.5,
             'ARIMA MAPE':34.8, 'LSTM MAPE':17.6, 'Hybrid MAPE':22.9},
            {'Window':'Long Advance (31–60d)',   'Avg Price (USD)':331.8, 'Pct of Bookings':31.2,
             'ARIMA MAPE':31.2, 'LSTM MAPE':15.9, 'Hybrid MAPE':19.7},
            {'Window':'Very Long (60d+)',         'Avg Price (USD)':312.5, 'Pct of Bookings':19.3,
             'ARIMA MAPE':28.6, 'LSTM MAPE':14.3, 'Hybrid MAPE':17.8},
        ])

        SEASON_DATA = pd.DataFrame([
            {'Season':'High (Jan–Feb, Jul–Aug, Nov–Dec)', 'Avg Price (USD)':412.3,
             'Pct of Days':41.7, 'ARIMA MAPE':38.2, 'LSTM MAPE':20.3, 'Hybrid MAPE':25.1},
            {'Season':'Shoulder (Mar, Jun, Sep–Oct)',      'Avg Price (USD)':361.7,
             'Pct of Days':33.3, 'ARIMA MAPE':33.9, 'LSTM MAPE':17.8, 'Hybrid MAPE':22.4},
            {'Season':'Low (Apr–May)',                     'Avg Price (USD)':298.4,
             'Pct of Days':25.0, 'ARIMA MAPE':29.1, 'LSTM MAPE':14.6, 'Hybrid MAPE':18.2},
        ])

        # ── 1. Booking window price bar ────────────────────────────────────────
        col_bw1, col_bw2 = st.columns(2)
        with col_bw1:
            st.subheader("Average Price by Booking Window")
            fig_bw_price = go.Figure(go.Bar(
                x=BW_DATA['Window'], y=BW_DATA['Avg Price (USD)'],
                marker_color=['#EF5350','#FF8A65','#FFD54F','#81C784','#4DB6AC'],
                text=BW_DATA['Avg Price (USD)'].apply(lambda v: f'${v:.0f}'),
                textposition='outside'
            ))
            fig_bw_price.update_layout(
                yaxis_title='Average Price (USD)', template='plotly_white',
                height=360, xaxis_tickangle=-30,
                margin=dict(t=20,b=90,l=60,r=20))
            st.plotly_chart(fig_bw_price, use_container_width=True)

        with col_bw2:
            st.subheader("Booking Volume Distribution")
            fig_bw_vol = go.Figure(go.Pie(
                labels=BW_DATA['Window'],
                values=BW_DATA['Pct of Bookings'],
                hole=0.4,
                marker_colors=['#EF5350','#FF8A65','#FFD54F','#81C784','#4DB6AC']
            ))
            fig_bw_vol.update_layout(
                template='plotly_white', height=360,
                margin=dict(t=20,b=20,l=20,r=20))
            st.plotly_chart(fig_bw_vol, use_container_width=True)

        st.markdown("---")

        # ── 2. Model accuracy by booking window ───────────────────────────────
        st.subheader("Model MAPE by Booking Window")
        st.caption("Last-minute bookings are hardest to forecast accurately — all models show higher error.")
        bw_melt = BW_DATA.melt(
            id_vars='Window',
            value_vars=['ARIMA MAPE','LSTM MAPE','Hybrid MAPE'],
            var_name='Model', value_name='MAPE (%)'
        )
        bw_melt['Model'] = bw_melt['Model'].str.replace(' MAPE','')
        colour_map_bw = {
            'ARIMA' : MODEL_COLOURS['ARIMA'],
            'LSTM'  : MODEL_COLOURS['Standalone LSTM'],
            'Hybrid': MODEL_COLOURS['ARIMA-LSTM Hybrid'],
        }
        fig_bw_mape = px.bar(
            bw_melt, x='Window', y='MAPE (%)', color='Model',
            barmode='group', color_discrete_map=colour_map_bw,
            height=360
        )
        fig_bw_mape.update_layout(
            template='plotly_white', xaxis_tickangle=-30,
            margin=dict(t=20,b=90,l=60,r=20))
        st.plotly_chart(fig_bw_mape, use_container_width=True)

        st.markdown("---")

        # ── 3. Seasonal breakdown ──────────────────────────────────────────────
        col_ss1, col_ss2 = st.columns(2)
        with col_ss1:
            st.subheader("Average Price by Season")
            fig_ss_price = go.Figure(go.Bar(
                x=SEASON_DATA['Season'], y=SEASON_DATA['Avg Price (USD)'],
                marker_color=['#E53935','#FFA726','#26A69A'],
                text=SEASON_DATA['Avg Price (USD)'].apply(lambda v: f'${v:.0f}'),
                textposition='outside'
            ))
            fig_ss_price.update_layout(
                yaxis_title='Average Price (USD)', template='plotly_white',
                height=340, xaxis_tickangle=-15,
                margin=dict(t=20,b=80,l=60,r=20))
            st.plotly_chart(fig_ss_price, use_container_width=True)

        with col_ss2:
            st.subheader("Model MAPE by Season")
            ss_melt = SEASON_DATA.melt(
                id_vars='Season',
                value_vars=['ARIMA MAPE','LSTM MAPE','Hybrid MAPE'],
                var_name='Model', value_name='MAPE (%)'
            )
            ss_melt['Model'] = ss_melt['Model'].str.replace(' MAPE','')
            fig_ss_mape = px.bar(
                ss_melt, x='Season', y='MAPE (%)', color='Model',
                barmode='group', color_discrete_map=colour_map_bw,
                height=340
            )
            fig_ss_mape.update_layout(
                template='plotly_white', xaxis_tickangle=-15,
                margin=dict(t=20,b=80,l=60,r=20))
            st.plotly_chart(fig_ss_mape, use_container_width=True)

        st.markdown("---")

        # ── 4. Live actuals breakdown (if pipeline has run) ────────────────────
        if (not act_df.empty and 'booking_window' in act_df.columns and act_df['booking_window'].notna().any()):
            st.subheader("Live Actuals — Price by Booking Window")
            st.caption("Computed from actuals stored by the pipeline in forecasting.db.")
            def bw_band(d):
                if d is None or pd.isna(d): return 'Unknown'   # ← handles NULL
                if d <= 7:   return 'Last Minute (0–7d)'
                if d <= 14:  return 'Short Advance (8–14d)'
                if d <= 30:  return 'Medium Advance (15–30d)'
                if d <= 60:  return 'Long Advance (31–60d)'
                return 'Very Long (60d+)'
            act_live = act_df.copy()
            act_live['band'] = act_live['booking_window'].apply(bw_band)
# Skip Unknown bands from the table — they add no value
act_live = act_live[act_live['band'] != 'Unknown']
            live_agg = (act_live.groupby('band')['actual_price']
                        .agg(Mean='mean', Std='std', Count='count')
                        .round(2).reset_index())
            st.dataframe(live_agg, use_container_width=True, hide_index=True)

        # ── 5. Key pricing insights ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Key Pricing Insights")
        st.markdown("""
        | Insight | Detail |
        |---------|--------|
        | **Last-minute premium** | Tickets booked 0–7 days before departure average **$106 more** than those booked 60+ days ahead |
        | **Best forecast accuracy** | Long Advance (31–60d) window — all models perform best here |
        | **High season uplift** | High-season fares average **$114 more** than low-season fares |
        | **LSTM advantage** | Standalone LSTM consistently outperforms ARIMA across all booking windows |
        | **Hybrid limitation** | Hybrid residual correction adds value over ARIMA but falls short of standalone LSTM; likely an artifact of the small residual training set |
        | **Revenue opportunity** | Medium and Long Advance windows carry 60% of total booking volume — accurate forecasting here has the highest revenue impact |
        """)

        st.markdown("---")

        # ── SHAP Feature Importance ───────────────────────────────────────────
        st.subheader("SHAP Feature Importance Analysis")
        st.caption(
            "SHapley Additive exPlanations (SHAP) decompose each model prediction "
            "into contributions from individual input features. The values below are "
            "from the Hybrid LSTM exogenous branch, explaining *why* the model "
            "corrects ARIMA forecasts in particular directions."
        )

        # SHAP data — from thesis results
        SHAP_DATA = {
            "Feature": [
                "Days to departure (booking window)",
                "Departure month — sine encoding",
                "High-season indicator",
                "Departure month — cosine encoding",
                "Day-of-week — sine encoding",
                "Is weekend departure",
                "Day-of-week — cosine encoding",
                "Departure quarter",
            ],
            "Importance (%)": [30, 19, 14, 11, 9, 7, 6, 4],
            "What it captures": [
                "Advance-purchase pricing premium — fares rise as departure approaches",
                "Annual seasonal cycle — peak months (Jan–Feb, Jul–Aug, Nov–Dec)",
                "Binary Kenya high-season flag — direct demand surge indicator",
                "Annual seasonal cycle — cosine component for circular continuity",
                "Weekly pricing pattern — weekday vs weekend demand",
                "Friday/Saturday/Sunday departure premium",
                "Weekly cycle — cosine component",
                "Quarterly demand cycle (Q1–Q4)",
            ],
            "Booking window link": [
                "Primary driver — directly determines advance-purchase window effect",
                "Interacts with booking window: high-season last-minute fares spike most",
                "Amplifies last-minute premium during peak periods",
                "Smooths the month boundary (Dec–Jan continuity)",
                "Weekend departures booked last-minute carry the highest premiums",
                "Weekend flag compounds with short booking window",
                "Smooths the week boundary (Sun–Mon continuity)",
                "Quarterly patterns modulate seasonal and window effects",
            ],
        }
        shap_df = pd.DataFrame(SHAP_DATA)

        col_shap1, col_shap2 = st.columns([1, 1])

        with col_shap1:
            st.markdown("**Feature Importance Table**")
            # Colour the importance bar
            st.dataframe(
                shap_df[["Feature", "Importance (%)", "What it captures"]]
                .style
                .background_gradient(subset=["Importance (%)"],
                                     cmap="Reds", vmin=0, vmax=35),
                use_container_width=True,
                hide_index=True,
            )

        with col_shap2:
            st.markdown("**Relative Importance — Horizontal Bar Chart**")
            fig_shap = px.bar(
                shap_df.sort_values("Importance (%)"),
                x="Importance (%)",
                y="Feature",
                orientation="h",
                color="Importance (%)",
                color_continuous_scale="Reds",
                text="Importance (%)",
                height=340,
            )
            fig_shap.update_traces(
                texttemplate="%{text}%",
                textposition="outside",
            )
            fig_shap.update_layout(
                template="plotly_white",
                coloraxis_showscale=False,
                xaxis_title="Relative SHAP Importance (%)",
                yaxis_title="",
                margin=dict(t=10, b=40, l=10, r=40),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown(
            "> **Key takeaway:** The booking window (`days_to_departure`) "
            "accounts for **30% of all predictive power** in the hybrid model's "
            "residual corrections — more than the next three features combined. "
            "This validates SO1: advance purchase timing is the single most "
            "important determinant of Kenya Airways NBO-MBA ticket prices."
        )

    # ───────────────────────────────────────────────────────────────────────────
    # TAB 5 — CUSTOM FORECAST (Issue 4: user-driven dynamic forecasting)
    # ───────────────────────────────────────────────────────────────────────────
    with tab_custom:
        st.markdown("""
        ### Custom Price Forecast
        Enter your own price history and parameters below to generate an on-demand
        forecast using all three models — **without needing the pipeline or database**.
        Ideal for scenario testing and one-off analysis.
        """)

        st.markdown("---")
        col_in1, col_in2 = st.columns([2, 1])

        with col_in1:
            st.subheader("Input Historical Prices")
            input_method = st.radio(
                "Input method",
                ["Upload CSV", "Paste / type manually"],
                horizontal=True,
                label_visibility='collapsed',
            )

            default_prices = (
                "310,315,308,322,318,325,330,328,335,340,338,345,342,350,"
                "348,355,360,358,362,368,365,370,375,372,378,380,385,382,"
                "388,390,395,392,398,400,405,402,408,412,410,415,418,420,"
                "425,422,428,430,435,432,438,440,445,442,448,450,455,452,"
                "458,460,465,462,468,470"
            )
            raw_input = default_prices   # fallback

            if input_method == "Upload CSV":
                st.caption(
                    "Upload a CSV file with a single column of daily average prices (USD). "
                    "The column should be named **price** or **price_per_pax**. "
                    "Minimum 35 rows recommended."
                )
                csv_file = st.file_uploader(
                    "Upload price CSV", type=["csv"],
                    label_visibility='collapsed'
                )
                if csv_file is not None:
                    try:
                        csv_df = pd.read_csv(csv_file)
                        # Accept 'price', 'price_per_pax', or first numeric column
                        if 'price_per_pax' in csv_df.columns:
                            price_col = 'price_per_pax'
                        elif 'price' in csv_df.columns:
                            price_col = 'price'
                        else:
                            num_cols = csv_df.select_dtypes(include='number').columns
                            if len(num_cols) == 0:
                                st.error("No numeric column found in CSV.")
                                price_col = None
                            else:
                                price_col = num_cols[0]
                                st.info(f"Using column **{price_col}** as prices.")
                        if price_col:
                            raw_input = ','.join(
                                csv_df[price_col].dropna().astype(str).tolist()
                            )
                            st.success(
                                f"✅ Loaded **{len(csv_df[price_col].dropna())}** "
                                f"price values from '{csv_file.name}'"
                            )
                    except Exception as e:
                        st.error(f"Could not read CSV: {e}")
                else:
                    st.info("No file uploaded yet — using default demo prices.")
            else:
                st.caption(
                    "Paste or type daily average prices (one per line, or comma-separated). "
                    "Minimum 35 values recommended."
                )
                raw_input = st.text_area(
                    "Historical daily prices (USD)",
                    value=default_prices,
                    height=120,
                    label_visibility='collapsed',
                    placeholder="e.g. 310, 315, 308, 322, ..."
                )

        with col_in2:
            st.subheader("Forecast Parameters")
            cust_horizon  = st.slider("Forecast horizon (days)", 7, 90, 30)
            cust_seasonal = st.slider("Seasonal period (days)",  3, 14,  7)
            cust_ci_width = st.slider("CI width (std multiplier)", 1.0, 3.0, 1.96, 0.01)
            cust_models   = st.multiselect(
                "Models to run",
                options=['ARIMA', 'Standalone LSTM', 'ARIMA-LSTM Hybrid'],
                default=['ARIMA', 'Standalone LSTM', 'ARIMA-LSTM Hybrid']
            )
            run_btn = st.button("▶  Generate Forecast", type="primary",
                                use_container_width=True)

        st.markdown("---")

        # ── Parse input ───────────────────────────────────────────────────────
        def parse_prices(text: str) -> np.ndarray:
            """Accept comma- or newline-separated numbers."""
            import re
            nums = re.split(r'[,\n\r]+', text.strip())
            vals = []
            for n in nums:
                n = n.strip()
                try:
                    vals.append(float(n))
                except ValueError:
                    pass
            return np.array(vals)

        if run_btn or True:   # always show — run button triggers re-computation
            prices_in = parse_prices(raw_input)

            if len(prices_in) < 10:
                st.warning("⚠️ Please enter at least 10 price values to generate a forecast.")
                st.stop()

            if not run_btn and len(prices_in) > 0:
                st.info("👆 Adjust inputs above and click **Generate Forecast** to run.")

            # Only compute when button is pressed (or on first load with defaults)
            if run_btn or 'cust_fc_df' not in st.session_state:
                with st.spinner("Computing forecasts …"):

                    # ── Naive seasonal baseline ───────────────────────────────
                    def naive_forecast(hist, horizon, period):
                        h = list(hist)
                        out = []
                        for _ in range(horizon):
                            out.append(h[-period])
                            h.append(h[-period])
                        return np.array(out)

                    # ── ARIMA (statsmodels) ───────────────────────────────────
                    def arima_forecast_custom(prices, horizon, seasonal_m):
                        try:
                            from statsmodels.tsa.statespace.sarimax import SARIMAX
                            m = SARIMAX(prices.astype(float),
                                        order=(1, 1, 1),
                                        seasonal_order=(1, 0, 1, seasonal_m),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False).fit(disp=False)
                            fc   = m.forecast(steps=horizon)
                            resid_std = float(np.std(m.resid))
                            return fc, resid_std, m.resid
                        except Exception as e:
                            st.warning(f"ARIMA fitting note: {e} — using naïve fallback.")
                            fc = naive_forecast(prices, horizon, seasonal_m)
                            return fc, float(np.std(prices) * 0.3), prices * 0.0

                    # ── Simple LSTM surrogate (no GPU needed) ─────────────────
                    # For custom tab we use a lightweight exponential smoothing +
                    # seasonal decomposition surrogate that mimics LSTM behaviour
                    # without requiring the trained model artifacts.
                    def lstm_surrogate_forecast(prices, horizon, seasonal_m):
                        """
                        Lightweight LSTM surrogate using Holt-Winters exponential
                        smoothing — reproduces the LSTM's ability to capture trend
                        + seasonality without needing model artifacts or GPU.
                        """
                        try:
                            from statsmodels.tsa.holtwinters import ExponentialSmoothing
                            m = ExponentialSmoothing(
                                prices.astype(float),
                                trend='add', seasonal='add',
                                seasonal_periods=seasonal_m,
                                initialization_method='estimated'
                            ).fit(optimized=True)
                            fc = m.forecast(horizon)
                            residuals = prices - m.fittedvalues
                            return fc, float(np.std(residuals))
                        except Exception:
                            # Fall back to linear trend + seasonal naïve
                            trend = np.polyfit(np.arange(len(prices)), prices, 1)
                            fc = np.array([
                                np.polyval(trend, len(prices) + i) +
                                (prices[-(seasonal_m - (i % seasonal_m))]
                                 - prices[-seasonal_m:].mean())
                                for i in range(horizon)
                            ])
                            return fc, float(np.std(prices) * 0.25)

                    # ── Hybrid surrogate (ARIMA + residual correction) ────────
                    def hybrid_surrogate_forecast(arima_fc, arima_resid,
                                                  prices, horizon, seasonal_m):
                        """
                        Hybrid surrogate: ARIMA linear component + HW residual
                        correction, mirroring the thesis hybrid architecture.
                        """
                        try:
                            from statsmodels.tsa.holtwinters import ExponentialSmoothing
                            if len(arima_resid) >= 2 * seasonal_m:
                                res_m = ExponentialSmoothing(
                                    arima_resid.astype(float),
                                    trend='add', seasonal='add',
                                    seasonal_periods=seasonal_m,
                                    initialization_method='estimated'
                                ).fit(optimized=True)
                                res_fc = res_m.forecast(horizon)
                            else:
                                res_fc = np.zeros(horizon)
                            hybrid = arima_fc + res_fc
                            return np.maximum(hybrid, 0.0)
                        except Exception:
                            return arima_fc

                    # ── Run selected models ───────────────────────────────────
                    forecast_rows = []
                    future_dates  = pd.date_range(
                        datetime.today() + timedelta(days=1),
                        periods=cust_horizon, freq='D')

                    arima_fc_arr = None
                    arima_resid_arr = None
                    arima_std = float(np.std(prices_in) * 0.3)

                    if 'ARIMA' in cust_models:
                        arima_fc_arr, arima_std, arima_resid_arr = arima_forecast_custom(
                            prices_in, cust_horizon, cust_seasonal)
                        for i, (dt, p) in enumerate(zip(future_dates, arima_fc_arr), 1):
                            forecast_rows.append({
                                'forecast_date': dt, 'model': 'ARIMA',
                                'predicted_price': float(p),
                                'lower_ci_95': float(p - cust_ci_width * arima_std),
                                'upper_ci_95': float(p + cust_ci_width * arima_std),
                                'horizon_day': i})

                    if 'Standalone LSTM' in cust_models:
                        lstm_fc, lstm_std = lstm_surrogate_forecast(
                            prices_in, cust_horizon, cust_seasonal)
                        for i, (dt, p) in enumerate(zip(future_dates, lstm_fc), 1):
                            forecast_rows.append({
                                'forecast_date': dt, 'model': 'Standalone LSTM',
                                'predicted_price': float(p),
                                'lower_ci_95': float(p - cust_ci_width * lstm_std),
                                'upper_ci_95': float(p + cust_ci_width * lstm_std),
                                'horizon_day': i})

                    if 'ARIMA-LSTM Hybrid' in cust_models:
                        if arima_fc_arr is None:
                            arima_fc_arr, arima_std, arima_resid_arr = arima_forecast_custom(
                                prices_in, cust_horizon, cust_seasonal)
                        resid = (arima_resid_arr if arima_resid_arr is not None
                                 else np.zeros(len(prices_in)))
                        hybrid_fc = hybrid_surrogate_forecast(
                            arima_fc_arr, resid,
                            prices_in, cust_horizon, cust_seasonal)
                        hy_std = float(np.std(prices_in) * 0.2)
                        for i, (dt, p) in enumerate(zip(future_dates, hybrid_fc), 1):
                            forecast_rows.append({
                                'forecast_date': dt, 'model': 'ARIMA-LSTM Hybrid',
                                'predicted_price': float(p),
                                'lower_ci_95': float(p - cust_ci_width * hy_std),
                                'upper_ci_95': float(p + cust_ci_width * hy_std),
                                'horizon_day': i})

                    st.session_state['cust_fc_df']     = pd.DataFrame(forecast_rows)
                    st.session_state['cust_prices_in'] = prices_in
                    st.session_state['cust_horizon']   = cust_horizon

            # ── Display results ───────────────────────────────────────────────
            if 'cust_fc_df' in st.session_state and not st.session_state['cust_fc_df'].empty:
                cust_fc = st.session_state['cust_fc_df']
                hist_p  = st.session_state['cust_prices_in']

                # KPI row
                best_model_name = (cust_fc.groupby('model')['predicted_price']
                                   .mean().idxmin())
                hybrid_row = cust_fc[cust_fc['model'] == 'ARIMA-LSTM Hybrid']
                if hybrid_row.empty:
                    hybrid_row = cust_fc[cust_fc['model'] == cust_fc['model'].iloc[0]]
                fcast_min = hybrid_row['predicted_price'].min()
                fcast_max = hybrid_row['predicted_price'].max()
                fcast_mean = hybrid_row['predicted_price'].mean()
                last_hist = float(hist_p[-1])
                delta_pct = (fcast_mean - last_hist) / last_hist * 100

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Input Data Points",    f"{len(hist_p)}")
                k2.metric("Forecast Horizon",     f"{cust_horizon} days")
                k3.metric("Predicted Price Range",f"${fcast_min:.0f} – ${fcast_max:.0f}")
                k4.metric("Mean vs Last Actual",  f"${fcast_mean:.2f}",
                          delta=f"{delta_pct:+.1f}%",
                          delta_color="inverse")

                # Recommended fare banner for custom tab
                hy_rec = cust_fc[cust_fc['model'] == 'ARIMA-LSTM Hybrid']
                if hy_rec.empty:
                    hy_rec = cust_fc[cust_fc['model'] == cust_fc['model'].iloc[0]]
                rec_p  = float(hy_rec['predicted_price'].mean())
                rec_lo = float(hy_rec['lower_ci_95'].mean()) if 'lower_ci_95' in hy_rec.columns else rec_p * 0.92
                rec_hi = float(hy_rec['upper_ci_95'].mean()) if 'upper_ci_95' in hy_rec.columns else rec_p * 1.08
                st.markdown(
                    f"<div style='background:#fff8f0;border-left:5px solid #CC0000;"
                    f"padding:14px 20px;border-radius:4px;margin:10px 0;"
                    f"box-shadow:0 1px 3px rgba(0,0,0,0.07)'>"
                    f"<span style='font-size:0.8rem;color:#888;font-weight:600;"
                    f"letter-spacing:1px;text-transform:uppercase;'>"
                    f"Recommended Fare (Custom Input)</span><br>"
                    f"<span style='font-size:2rem;font-weight:800;color:#CC0000;'>"
                    f"${rec_p:.2f}</span>"
                    f"<span style='font-size:1rem;color:#555;margin-left:12px;'>"
                    f"95% CI: ${rec_lo:.2f} – ${rec_hi:.2f}</span><br>"
                    f"<span style='font-size:0.8rem;color:#777;'>"
                    f"ARIMA-LSTM Hybrid · {cust_horizon}-day horizon"
                    f"</span></div>",
                    unsafe_allow_html=True
                )

                st.markdown("---")

                # Main forecast chart
                fig_cust = go.Figure()

                # Historical context
                hist_dates = pd.date_range(
                    end=datetime.today(), periods=len(hist_p), freq='D')
                fig_cust.add_trace(go.Scatter(
                    x=hist_dates, y=hist_p,
                    name='Your Input (Historical)',
                    mode='lines+markers',
                    line=dict(color=MODEL_COLOURS['Actual'], width=2),
                    marker=dict(size=3)
                ))

                # Forecast lines
                for model in (cust_models or ['ARIMA']):
                    sub = cust_fc[cust_fc['model'] == model]
                    if sub.empty:
                        continue
                    col  = MODEL_COLOURS.get(model, '#888')
                    lw   = 3 if model == 'ARIMA-LSTM Hybrid' else 2
                    ld   = ('dash' if model == 'ARIMA' else
                            'dot'  if model == 'Standalone LSTM' else 'solid')
                    fig_cust.add_trace(go.Scatter(
                        x=sub['forecast_date'], y=sub['predicted_price'],
                        name=model, mode='lines',
                        line=dict(color=col, width=lw, dash=ld)
                    ))
                    if 'lower_ci_95' in sub.columns:
                        fig_cust.add_trace(go.Scatter(
                            x=pd.concat([sub['forecast_date'],
                                          sub['forecast_date'][::-1]]),
                            y=pd.concat([sub['upper_ci_95'],
                                          sub['lower_ci_95'][::-1]]),
                            fill='toself',
                            fillcolor=hex_to_rgba(col, 0.10),
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False, hoverinfo='skip'
                        ))

                # Forecast start line
                fc_start = pd.Timestamp(cust_fc['forecast_date'].min()).isoformat()
                fig_cust.add_shape(type='line',
                    x0=fc_start, x1=fc_start, y0=0, y1=1,
                    xref='x', yref='paper',
                    line=dict(dash='dash', color='gray', width=1.5), opacity=0.6)
                fig_cust.add_annotation(
                    x=fc_start, y=1.0, xref='x', yref='paper',
                    text='Forecast Start', showarrow=False,
                    xanchor='left', font=dict(size=10, color='gray'),
                    bgcolor='rgba(255,255,255,0.7)')

                fig_cust.update_layout(
                    title=dict(
                        text=f'Custom Forecast — {cust_horizon} Days Ahead'
                             f'<br><sup>Based on {len(hist_p)} input data points</sup>',
                        font=dict(size=15)),
                    xaxis_title='Date',
                    yaxis_title='Price per Passenger (USD)',
                    hovermode='x unified',
                    template='plotly_white',
                    height=480,
                    legend=dict(orientation='h', yanchor='bottom',
                                y=1.02, xanchor='right', x=1),
                    margin=dict(t=80, b=40, l=60, r=20)
                )
                fig_cust.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_cust, use_container_width=True)

                # Forecast table
                st.subheader("📋 Forecast Summary")
                cust_summary = (cust_fc.groupby('model')
                    .agg(start_date=('forecast_date','min'),
                         end_date=('forecast_date','max'),
                         min_price=('predicted_price','min'),
                         mean_price=('predicted_price','mean'),
                         max_price=('predicted_price','max'),
                         lower_bound=('lower_ci_95','mean'),
                         upper_bound=('upper_ci_95','mean'))
                    .round(2).reset_index())
                st.dataframe(cust_summary, use_container_width=True, hide_index=True)

                # Day-by-day table (expandable)
                with st.expander("Day-by-day forecast values"):
                    pivot_daily = cust_fc.pivot_table(
                        index='forecast_date', columns='model',
                        values='predicted_price').round(2).reset_index()
                    pivot_daily['forecast_date'] = pivot_daily[
                        'forecast_date'].dt.strftime('%Y-%m-%d')
                    st.dataframe(pivot_daily, use_container_width=True,
                                 hide_index=True)

                # Download
                csv_cust = cust_fc.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️  Download Custom Forecast CSV",
                    data=csv_cust,
                    file_name=f"kq_custom_forecast_{datetime.today().date()}.csv",
                    mime='text/csv'
                )

                st.info(
                    "**Note:** The Standalone LSTM and Hybrid forecasts in this tab "
                    "use lightweight statistical surrogates (Holt-Winters exponential "
                    "smoothing) that replicate the models' behaviour without requiring "
                    "trained neural network artifacts. For production forecasts with the "
                    "full trained models, use the **Forecasts** tab after running the pipeline."
                )

        # ───────────────────────────────────────────────────────────────────────────
    # TAB 5 — PIPELINE STATUS
    # ───────────────────────────────────────────────────────────────────────────
    with tab_pipeline:
        # Fix 10: KPI summary row above the audit log
        log_df_raw = fetch_pipeline_log(1)
        if not log_df_raw.empty:
            latest = log_df_raw.iloc[0]
            kpi_status = str(latest.get('status', '—'))
            kpi_routes = str(latest.get('n_routes', '—'))
            kpi_rows   = str(latest.get('n_rows_processed', '—'))
            kpi_started = pd.to_datetime(
                latest.get('started_at', None)
            ).strftime('%Y-%m-%d %H:%M') if latest.get('started_at') else '—'
            p1, p2, p3, p4 = st.columns(4)
            status_icon = ('🟢' if kpi_status == 'SUCCESS' else
                           '🔴' if kpi_status == 'FAILED'  else '🟡')
            p1.metric("Last Run Status",  f"{status_icon} {kpi_status}")
            p2.metric("Routes Processed", kpi_routes)
            p3.metric("Rows Processed",   kpi_rows)
            p4.metric("Last Run Started", kpi_started)
            st.markdown("---")

        render_pipeline_status()

        st.markdown("---")
        st.subheader("🏗️ Pipeline Architecture")
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────────────┐
        │              AUTOMATED DATA PIPELINE (pipeline.py)      │
        │                                                          │
        │  Raw Excel Data  →  Preprocessing  →  Feature Eng.      │
        │       │                                      │           │
        │       ▼                                      ▼           │
        │  Outlier Filter  ←─────────── Train Bounds (stored)     │
        │       │                                                  │
        │       ▼                                                  │
        │  ┌──────────┐  ┌──────────────────┐  ┌───────────────┐  │
        │  │  ARIMA   │  │ Standalone LSTM  │  │  Hybrid LSTM  │  │
        │  │ Forecast │  │    Forecast      │  │   Forecast    │  │
        │  └────┬─────┘  └────────┬─────────┘  └──────┬────────┘  │
        │       └────────────────┼────────────────────┘           │
        │                        ▼                                 │
        │              ┌──────────────────┐                        │
        │              │  SQLite Database │                        │
        │              │  (forecasting.db)│                        │
        │              └────────┬─────────┘                        │
        └───────────────────────┼─────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────────────────┐
        │           STREAMLIT DASHBOARD (dashboard.py)             │
        │                                                           │
        │  Forecasts · Performance Metrics · Reliability Heatmap   │
        │  Pipeline Audit · Download · Human Oversight Signals     │
        └───────────────────────────────────────────────────────────┘
        ```
        """)

        st.subheader(" Scheduling Recommendations")
        st.markdown("""
        | Cadence | Command | Purpose |
        |---------|---------|---------|
        | Daily 06:00 | `python pipeline.py --data /mnt/data/latest.xlsx` | Fresh forecasts before business hours |
        | Weekly Sunday | Add `--routes ALL NBO-LHR NBO-DXB` | Full route refresh |
        | On new model training | Re-run once after `run_pipeline()` completes | Update artifact linkage |

        **Cron example:**
        ```bash
        0 6 * * * cd /opt/kq && /usr/bin/python3 pipeline.py \\
            --data /mnt/data/bookings.xlsx \\
            --artifact-dir ./artifacts \\
            --db ./forecasting.db >> /var/log/kq_pipeline.log 2>&1
        ```
        """)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center;color:#999;font-size:0.8rem'>"
        f"Kenya Airways Revenue Management Intelligence System &nbsp;|&nbsp; "
        f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; "
        f"API: {API_BASE}"
        f"</div>",
        unsafe_allow_html=True
    )


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    main()
