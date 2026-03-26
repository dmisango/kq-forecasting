# Kenya Airways Air Ticket Price Forecasting
## MSc Data Science & Analytics — Strathmore University

---

## Project Identity

| Field | Detail |
|---|---|
| **Student** | Dorcas Engasia Misango |
| **Admission No.** | 100235 |
| **Degree** | MSc Data Science and Analytics |
| **Institution** | Strathmore University, Nairobi, Kenya |
| **Supervisor** | Dr. Joseph Sevilla |
| **Thesis Title** | Integrating Statistical and Deep Learning Approaches for Airline Price Forecasting: A Hybrid ARIMA-LSTM Framework Applied to Kenya Airways Prices |

---

## What This Project Does

This project builds and evaluates three forecasting models for Kenya Airways ticket prices on the Nairobi–Mombasa (NBO-MBA) route using 2016–2020 historical booking data. The three models are:

1. **ARIMA** — a classical statistical model that captures linear trend and seasonality
2. **Standalone LSTM** — a deep learning model that captures non-linear temporal patterns
3. **ARIMA-LSTM Hybrid** — combines ARIMA's linear forecast with an LSTM that learns the ARIMA residuals (non-linear correction)

The best-performing model (Standalone LSTM, MAE = 5.98 USD, MAPE = 18.41%) is deployed as a Streamlit dashboard and FastAPI REST service.

---

## Repository Structure

```
kq-forecasting/
│
├── CPV Data NBO-MBA 2016-2020.xlsx          ← Raw data (NOT included in repo — required to run)
│
├── air_ticket_price_forecasting_thesis.py   ← MAIN: end-to-end training pipeline
│                                               Sections 0–20, 2,547 lines
│                                               Run ONCE to train models and save artifacts
│
├── pipeline.py                              ← inference pipeline (708 lines)
│                                               Loads data, runs trained models,
│                                               writes forecasts to forecasting.db
│
├── api.py                                   ← FastAPI REST service (800 lines)
│                                               6 endpoints — serves forecasts, metrics,
│                                               actuals, and pipeline status over HTTP
│
├── dashboard.py                             ← Streamlit web dashboard (1,797 lines)
│                                               Interactive UI for revenue analysts
│                                               Consumes data exclusively through api.py
│
├── forecasting.db                           ← SQLite database (auto-created by pipeline.py)
│                                               Tables: forecasts, actuals,
│                                               performance_metrics, pipeline_runs
│
├── artifacts/                               ← Trained model artifacts (auto-created)
│   ├── arima_model.pkl
│   ├── standalone_lstm_model.keras
│   ├── hybrid_lstm_model.keras
│   ├── price_scaler.pkl
│   ├── residual_scaler.pkl
│   ├── exog_scaler.pkl
│   ├── label_encoders.pkl
│   ├── model_meta.pkl
│   └── future_forecast_all_models.csv
│
├── README.md                                ← This file
└── PROJECT_SUMMARY.md                       ← Comprehensive thesis context for LLMs
```

---

## System Architecture

```
CPV Data NBO-MBA 2016-2020.xlsx
           │
           │  (run once)
           ▼
air_ticket_price_forecasting_thesis.py
  - cleans and preprocesses data
  - trains ARIMA, Standalone LSTM, Hybrid LSTM
  - evaluates on held-out test set
  - runs Diebold-Mariano significance tests
  - saves trained model artifacts to /artifacts/
  - saves future forecasts to CSV
           │
           ▼
/artifacts/  +  future_forecast_all_models.csv
           │
           │  (runs daily via cron/scheduler)
           ▼
pipeline.py
  - loads new booking data
  - loads trained artifacts
  - runs inference for each route
  - writes to forecasting.db
           │
           ▼
forecasting.db  (sole data store — only api.py reads from it)
           │
           ▼
api.py  (FastAPI, port 8000 — sole gateway to the database)
  GET  /health
  GET  /routes
  GET  /forecast/latest
  POST /forecast/custom
  GET  /performance
  GET  /pipeline/runs
  GET  /data/actuals
  GET  /data/summary
           │
     ┌─────┴──────┐
     ▼            ▼
dashboard.py   External systems
(Streamlit,    (Excel, mobile,
 port 8501)    revenue mgmt)
```

---

## How to Run

### Prerequisites

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Install all dependencies
pip install tensorflow pandas numpy scikit-learn statsmodels pmdarima \
            joblib matplotlib seaborn shap scipy \
            fastapi uvicorn requests streamlit plotly
```

### Step 1 — Train models (run once)

```bash
# Update DATA_PATH in air_ticket_price_forecasting_thesis.py line ~76 first
python air_ticket_price_forecasting_thesis.py
```

Takes 20–60 minutes depending on hardware. Produces `/artifacts/` folder and saves all plots as PNG files.

### Step 2 — Run daily pipeline

```bash
python pipeline.py --data "CPV Data NBO-MBA 2016-2020.xlsx" --artifact-dir artifacts
```

### Step 3 — Start API (Terminal 2)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
# or if uvicorn not on PATH:
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Verify at: http://localhost:8000/health  
Interactive docs at: http://localhost:8000/docs

### Step 4 — Start dashboard (Terminal 3)

```bash
streamlit run dashboard.py
```

Opens automatically in browser at http://localhost:8501

---

## Key Results (Test Set — NBO-MBA Route)

| Model | MAE (USD) | RMSE (USD) | MAPE (%) | R² |
|---|---|---|---|---|
| Naïve (Seasonal) | 28.49 | 31.65 | 78.04 | 0.12 |
| ARIMA | 14.87 | 17.62 | 35.54 | 0.61 |
| **Standalone LSTM** | **5.98** | **7.11** | **18.41** | **0.89** |
| ARIMA-LSTM Hybrid | 8.43 | 10.22 | 22.17 | 0.84 |

**Standalone LSTM achieves a 79% MAE reduction over the naïve baseline.**

Diebold-Mariano test confirms all three models are statistically significantly better than the naïve baseline (p < 0.01). Standalone LSTM is significantly better than ARIMA (p < 0.001). Hybrid vs Standalone LSTM difference is not statistically significant (p = 0.156).

---

## Model Configuration

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Train / Val / Test split | 70% / 15% / 15% |
| ARIMA seasonal period | 7 (weekly) |
| SA-LSTM sequence length | 30 days |
| Hybrid LSTM sequence length | 14 days |
| Forecast horizon | 30 days |
| SA-LSTM architecture | BiLSTM(64) → LSTM(32) → Dense(1) |
| SA-LSTM dropout | 0.20 |
| SA-LSTM learning rate | 1e-3 |
| SA-LSTM epochs | 150 (early stopping patience=20) |
| SA-LSTM batch size | 32 |
| Hybrid LSTM architecture | Dual-input: residuals + exogenous |
| Hybrid dropout | 0.20 |
| Hybrid learning rate | 5e-4 |
| Hybrid epochs | 200 (patience=30) |
| Hybrid L2 regularisation | 1e-4 |
| Confidence intervals | 95% empirical (MC Dropout for future forecasts) |

---

## Data Description

**File:** `CPV Data NBO-MBA 2016-2020.xlsx`  
**Route:** Nairobi (NBO) to Mombasa (MBA) — Kenya Airways regional domestic route  
**Period:** January 2016 – December 2020 (1,826 days)  
**Records:** ~15,000 individual booking transactions  
**Target variable:** `price_per_pax` — average fare per passenger in USD  
**Key columns:**
- `Date of issue day` — booking/issue date
- `Departure date` — flight departure date
- `Flight-Flight pair` — route code
- `price_per_pax` — derived target: total revenue / passengers
- `booking_window` — days between issue and departure
- `departure_month`, `departure_day_of_week` — temporal features

**COVID-19 impact:** Data includes the COVID-19 disruption period (March 2020 onward). A vertical marker at 2020-03-01 is added to all visualisations. The test set partially overlaps with this period, which elevates error metrics for all models.

---

## Thesis Objectives

**General Objective:** To assess the performance of ARIMA, LSTM and hybrid ARIMA-LSTM models in forecasting airline ticket prices for Kenya Airways.

**Specific Objectives:**
1. Identify and analyse key determinants influencing price variability in Kenya Airways regional routes
2. Evaluate individual performance of ARIMA and LSTM models
3. Develop and test a hybrid ARIMA-LSTM forecasting model
4. Compare and validate hybrid vs standalone models using statistical significance testing

---

## Research Questions

1. What are the key determinants of price variability on Kenya Airways regional routes?
2. How accurately do ARIMA and LSTM models individually forecast ticket prices?
3. Does combining ARIMA and LSTM into a hybrid model improve forecast accuracy?
4. Is the hybrid model's performance improvement statistically significant?

---

## Dependencies (Full List)

```
tensorflow>=2.10
pandas>=1.5
numpy>=1.23
scikit-learn>=1.1
statsmodels>=0.13
pmdarima>=2.0
joblib>=1.2
matplotlib>=3.6
seaborn>=0.12
shap>=0.41
scipy>=1.9
fastapi>=0.95
uvicorn>=0.21
requests>=2.28
streamlit>=1.22
plotly>=5.13
openpyxl>=3.0        # for reading Excel files
```

---

## Known Issues and Notes

1. `fillna(method='ffill')` is deprecated in pandas ≥ 2.0. Replace with `.ffill().bfill()` in:
   - `pipeline.py` line 326
   - `air_ticket_price_forecasting_thesis.py` lines 364 and 2224

2. The `performance_metrics` table in `forecasting.db` only populates once forecast dates have passed and actual prices are recorded. Until then, the dashboard and API fall back to the hardcoded thesis test-set results.

3. The hybrid LSTM walk-forward cross-validation is intentionally omitted (computationally prohibitive). This is flagged as a further-work item.

4. `DATA_PATH` in `air_ticket_price_forecasting_thesis.py` line ~76 must be updated to the actual Excel file location before running.

---

## Output Files Generated by the Training Pipeline

| File | Description |
|---|---|
| `Price_Series_Rolling_Statistics.png` | Price series with 30-day rolling mean and std |
| `stl_decomposition.png` | STL seasonal decomposition (trend, seasonal, residual) |
| `acf_pacf.png` | ACF and PACF plots for ARIMA order selection |
| `three_way_comparison.png` | Full timeline + per-model test-set comparison panels |
| `metric_comparison.png` | Bar charts: MAE, RMSE, MAPE, R² across all models |
| `cv_mae_comparison.png` | Walk-forward CV MAE box plots |
| `shap_feature_importance.png` | SHAP feature importance for hybrid LSTM exogenous branch |
| `future_forecast_all_models.png` | 30-day ahead forecast with MC Dropout CIs |
| `Standalone_LSTM_training.png` | Training / validation loss and MAE curves |
| `ARIMA-LSTM_Hybrid_training.png` | Training / validation loss and MAE curves |
| `future_forecast_all_models.csv` | Numeric forecast values for all models |