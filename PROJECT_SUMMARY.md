# PROJECT SUMMARY — FOR LLM THESIS GENERATION
## Kenya Airways Air Ticket Price Forecasting

---

## HOW TO USE THIS DOCUMENT

This document provides everything an LLM needs to generate Chapters 3 through 7 of the thesis. Read it fully before writing any chapter. All numerical results, model configurations, architectural decisions, and methodological rationale are documented here. Do not invent numbers — use only the values stated in this document.

When generating chapters, also read the following files if they are provided:
- `air_ticket_price_forecasting_thesis.py` — the complete training code with inline THESIS NOTEs explaining every architectural decision
- `pipeline.py` — the production inference pipeline
- `api.py` — the REST API deployment layer
- `dashboard.py` — the Streamlit dashboard

---

## PART 1 — THESIS IDENTITY AND STRUCTURE

**Title:** Integrating Statistical and Deep Learning Approaches for Airline Price Forecasting: A Hybrid ARIMA-LSTM Framework Applied to Kenya Airways Prices

### Expected Chapter Structure

- **Chapter 1** — Introduction (from proposal — do not regenerate)
- **Chapter 2** — Literature Review (from proposal — do not regenerate)
- **Chapter 3** — Methodology
- **Chapter 4** — System Design and Architecture
- **Chapter 5** — System Development, Testing and Validation
- **Chapter 6** — Discussion of Results
- **Chapter 7** — Conclusions, Recommendations and Future Work
- **References**
- **Appendices**

---

## PART 2 — RESEARCH OBJECTIVES AND QUESTIONS

### General Objective
To assess the performance of ARIMA, LSTM and hybrid ARIMA-LSTM models in forecasting airline ticket prices for Kenya Airways.

### Specific Objectives
- SO1: Identify and analyse key determinants influencing price variability in Kenya Airways regional routes
- SO2: Evaluate individual performance of ARIMA and LSTM models
- SO3: Develop and test a hybrid ARIMA-LSTM forecasting model
- SO4: Compare and validate hybrid vs standalone models using statistical significance testing

### Research Questions
- SRQ1: What are the key determinants of price variability on Kenya Airways regional routes?
- SRQ2: How accurately do ARIMA and LSTM models individually forecast ticket prices?
- SRQ3: Does combining ARIMA and LSTM into a hybrid model improve forecast accuracy?
- SRQ4: Is the hybrid model's performance improvement statistically significant?

---

## PART 3 — DATASET

### Source and Scope
- **File:** CPV Data NBO-MBA 2016-2020.xlsx
- **Airline:** Kenya Airways (KQ)
- **Route:** Nairobi (NBO) → Mombasa (MBA) — domestic regional route, ~1 hour flight
- **Period:** January 2016 – December 2020 (5 years, 1,826 days)
- **Raw records:** approximately 949,360, individual booking transactions
- **After daily aggregation:** 2,535 daily price observations

### Key Variables
| Column | Description |
|---|---|
| `Date of issue day` | Date the booking/ticket was issued |
| `Departure date` | Scheduled flight departure date |
| `Flight-Flight pair` | Route code (NBO-MBA) |
| `price_per_pax` | Target variable: average fare per passenger (USD) |
| `booking_window` | Days between issue date and departure date |
| `departure_month` | Month of departure (1–12) |
| `departure_day_of_week` | Day of week of departure (0=Monday) |
| `Flown CPV` | Total revenue for the booking |
| `Flown seg pax` | Number of passengers |

### Derived Features (added during preprocessing)
- `departure_month_sin`, `departure_month_cos` — cyclical annual encoding
- `departure_day_of_week_sin`, `departure_day_of_week_cos` — cyclical weekly encoding
- `is_high_season` — binary: 1 for Jan–Feb, Jul–Aug, Nov–Dec (Kenya peak demand)
- `is_weekend_departure` — binary: 1 for Friday, Saturday, Sunday
- `departure_quarter` — 1–4
- `days_to_departure` — same as booking_window
- `booking_category_ord` — ordinal encoding of booking window band

### Descriptive Statistics (full dataset, pre-split)
| Statistic | Value |
|---|---|
| Mean price | USD 53.37 |
| Median price | USD 40.50 |
| Std deviation | USD 47.64 |
| Minimum | USD 0.0 |
| Maximum | USD 4,644.07 |
| 25th percentile | USD 19.66 |
| 75th percentile | USD 74.84 |

**FIndings:** 
  • Mean price USD 53.37 vs median USD 40.50 — right-skewed distribution (skewness = 6.321)
  • Coefficient of variation = 89.3% — high price volatility
  • IQR = USD 55.18 (Q1=19.656666666666666 — Q3=74.83624999999999)
  • COVID-19 mean price change: -28.3% (pre=54.28 → post=38.89 USD)

### Booking Window Distribution
| Window | Mean Price (USD) | % of Bookings |
|---|---|---|
| Last Minute (0–7 days) | 55.31 | 53.90% |
| Short Advance (8–14 days) | 59.55 | 14.50% |
| Medium Advance (15–30 days) | 55.47 | 12.50% |
| Long Advance (31–60 days) | 45.78 | 8.90% |
| Very Long (60+ days) | 38.26 | 10.10% |

**Finding:** Last-minute premium = USD 17.05 (44.6% more than Very Long advance)

### Seasonal Distribution
| Season | Months | Mean Price (USD) | % of Days |
|---|---|---|---|
| High Season | Jan–Feb, Jul–Aug, Nov–Dec | 53.70 | 52.3% |
| Shoulder Season | Mar, Jun, Sep–Oct | 52.07 | 33.1% |
| Low Season | Apr–May | 55.12 | 14.7% |

**Finding:** High-season premium = USD -1.42 (2.6% below low season).

### COVID-19 Impact
- COVID-19 travel disruptions began March 2020
- Marker date: 2020-03-01
- The test set (last 15% of the chronological series) partially overlaps with COVID-19, which increases forecast errors for all models
- All plots include a vertical dashed line at 2020-03-01

---

## PART 4 — DATA PREPROCESSING

### Pipeline (in order, all in `air_ticket_price_forecasting_thesis.py`)

1. **Load and inspect** — read Excel, check dtypes, missing values, duplicates
2. **Date parsing** — convert `Date of issue day` and `Departure date` to datetime
3. **Target derivation** — `price_per_pax = Flown CPV / Flown seg pax`
4. **Booking window calculation** — `(Departure date - Date of issue day).days`
5. **Feature engineering** — add cyclical encodings, season flags, quarter
6. **Outlier removal** — IQR×3 bounds computed on training set only, applied to all splits. Bounds: lower = Q1 − 3×IQR, upper = Q3 + 3×IQR
7. **Categorical encoding** — LabelEncoder fitted on training set only, applied to validation and test
8. **Chronological train/val/test split** — 70% / 15% / 15% by date (no shuffling)
9. **Daily aggregation** — group by departure date, take median price per day
10. **Forward-fill missing dates** — `ffill().bfill()` for any gaps in the daily series
11. **Scaling** — MinMaxScaler fitted on training set, applied to all splits

### Data Leakage Prevention
- All scalers, encoders, and outlier bounds are fitted exclusively on the training set
- Validation and test sets are transformed using training-fitted parameters
- The chronological split ensures no future information contaminates model training

### Stationarity Analysis
The Augmented Dickey-Fuller (ADF) test and KPSS test are both applied:
- **Raw series:** non-stationary (ADF fails to reject unit root)
- **First-differenced series:** stationary (ADF rejects at p < 0.01)
- **Conclusion:** series is I(1) — integrated of order 1 — justifying d=1 in ARIMA

---

## PART 5 — MODEL ARCHITECTURES AND TRAINING

### 5.1 Naïve Seasonal Baseline
- **Formula:** ŷ(t) = y(t − s) where s = 7 (weekly seasonal period)
- **Purpose:** minimum performance benchmark that all learned models must exceed
- **Implementation:** `naive_seasonal_forecast()` in thesis file Section 11

### 5.2 ARIMA Model
- **Selection method:** auto_arima (pmdarima) with AIC criterion, stepwise search
- **Search space:** p ∈ [0,5], q ∈ [0,5], d=1 (from stationarity tests), seasonal_m=7
- **Selected order:** determined by auto_arima on training data (printed at runtime)
- **Seasonal order:** SARIMAX seasonal component included if improvement > AIC threshold
- **Training set:** 70% of chronological data
- **Validation:** used for model selection only, not for parameter estimation
- **Diagnostics:** Ljung-Box test for residual autocorrelation, normality Q-Q plot
- **Forecast method:** recursive multi-step (re-fitted on train+val for test evaluation)

### 5.3 Standalone LSTM
- **Architecture:** Bidirectional LSTM(64 units) → LSTM(32 units) → Dense(1)
- **Input:** sequences of 30 consecutive daily prices (look-back window = 30 days)
- **Dropout:** 0.20 after each LSTM layer
- **Optimiser:** Adam, learning rate = 1e-3
- **Loss function:** Mean Squared Error
- **Batch size:** 32
- **Epochs:** 150 maximum
- **Early stopping:** patience = 20 on validation loss
- **Learning rate reduction:** ReduceLROnPlateau, patience = 10, factor = 0.5
- **Scaling:** MinMaxScaler fitted on training prices
- **Forecasting method:** recursive — each predicted price is appended to the input window for the next step
- **Hyperparameter tuning:** random search over {units: [32,64,128], dropout: [0.1,0.2,0.3], lr: [1e-3, 5e-4, 1e-4], batch: [16,32,64]}

**Why Bidirectional LSTM as first layer:** Bidirectional processing captures both forward (trend) and backward (correction) temporal dependencies in the price series, which is particularly useful for a series with strong weekly seasonality where context from nearby future values (within the look-back window) aids current prediction.

**Why 30-day look-back:** Captures slightly more than 4 complete weekly seasonal cycles, allowing the model to learn both the weekly pattern and month-level trend. Tested against 14 and 60 days; 30 days gave best validation MAE.

### 5.4 Hybrid ARIMA-LSTM Model
- **Architecture:** Dual-input LSTM — Input 1: ARIMA residuals; Input 2: exogenous temporal features
- **Hybrid formula:** ŷ_hybrid(t) = ŷ_ARIMA(t) + ê_LSTM(t)
  - where ê_LSTM(t) is the LSTM's prediction of the ARIMA residual at time t
- **Input 1 (residuals):** sequences of 14 consecutive ARIMA training residuals
- **Input 2 (exogenous):** 8 temporal features for each of the 14 days in the window
- **LSTM layers:** LSTM(64) → LSTM(32) on residual branch; merged with Dense projection of exogenous
- **Dropout:** 0.20
- **Optimiser:** Adam, learning rate = 5e-4 (lower than SA-LSTM to prevent overfitting residuals)
- **L2 regularisation:** 1e-4 on kernel weights
- **Epochs:** 200 (patience = 30)
- **Batch size:** 32

**Why 14-day look-back for hybrid (not 30):** ARIMA residuals are mean-zero, shorter-memory signals whose autocorrelation typically decays within 7–14 lags (one or two weekly seasonal cycles). Using the same 30-day window as the standalone LSTM forces the network to process 30 near-zero lags, diluting the informative recent signal with stale noise. A 14-day window captures the relevant weekly residual pattern while keeping the input compact. This is consistent with recommendations for residual-correction LSTMs (Zhang 2003, Khashei & Bijari 2011).

**Why 8 exogenous features (not all features):** The original implementation used all 30+ numeric columns from the booking DataFrame, including booking-side aggregations (booking_window, booking_category) averaged by departure date. These averages are noisy (they aggregate heterogeneous bookings) and caused the hybrid to overfit the small residual training set (~1,000 sequences), adding systematic noise rather than reducing it. The curated 8 features are uniquely determined by the departure date itself — they are deterministic, low-variance, and directly encode the seasonality that ARIMA misses.

**The 8 exogenous features:**
1. `departure_month_sin` — annual seasonality (sine component)
2. `departure_month_cos` — annual seasonality (cosine component)
3. `departure_day_of_week_sin` — weekly seasonality (sine)
4. `departure_day_of_week_cos` — weekly seasonality (cosine)
5. `is_high_season` — binary Kenya demand peak indicator
6. `is_weekend_departure` — binary Friday/Saturday/Sunday premium flag
7. `departure_quarter` — quarterly demand cycle
8. `days_to_departure` — advance purchase window (booking incentive)

### 5.5 Walk-Forward Cross-Validation
- **Method:** TimeSeriesSplit with 5 folds
- **Applied to:** Naïve, ARIMA, Standalone LSTM
- **NOT applied to Hybrid:** computationally prohibitive (requires refitting ARIMA + LSTM in each fold). Flagged as a further-work item.
- **Metric reported:** MAE per fold, summarised as mean ± std across folds

### 5.6 Uncertainty Quantification
- **Method:** Monte Carlo Dropout (MC Dropout)
- **Application:** future forecast only (not used in test-set evaluation)
- **Samples:** 100 forward passes with dropout active at inference time
- **Output:** 95% confidence intervals for Standalone LSTM and Hybrid future forecasts
- **Reference:** Gal & Ghahramani (2016) — dropout as a Bayesian approximation

---

## PART 6 — EVALUATION METRICS

All metrics are computed on the held-out test set (last 15% of chronological data).

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | (1/n) Σ|y − ŷ| | Average absolute error in USD — directly interpretable |
| RMSE | √((1/n) Σ(y − ŷ)²) | Penalises large errors more than MAE |
| MAPE | (100/n) Σ|y − ŷ|/y | Percentage error — scale-independent |
| R² | 1 − SS_res/SS_tot | Proportion of variance explained (1.0 = perfect) |

### Diebold-Mariano Statistical Significance Test
The DM test (Harvey, Leybourne & Newbold 1997, small-sample correction applied) tests whether two forecasting models have statistically equal predictive accuracy.

- H₀: equal predictive accuracy
- H₁: one model is significantly more accurate
- Negative DM statistic: left/first model is more accurate
- Significance threshold: α = 0.05

**Results:**

| Comparison | DM Statistic | p-value | Conclusion |
|---|---|---|---|
| ARIMA vs Naïve | −3.21 | 0.002 | ARIMA significantly better |
| SA-LSTM vs Naïve | −5.84 | <0.001 | SA-LSTM significantly better |
| Hybrid vs Naïve | −4.62 | <0.001 | Hybrid significantly better |
| SA-LSTM vs ARIMA | −4.11 | <0.001 | SA-LSTM significantly better |
| Hybrid vs ARIMA | −2.87 | 0.005 | Hybrid significantly better |
| Hybrid vs SA-LSTM | +1.43 | 0.156 | No significant difference |

---

## PART 7 — TEST-SET RESULTS (THE DEFINITIVE NUMBERS)

These are the exact numbers to use in all chapters. Do not alter them.

| Model | MAE (USD) | RMSE (USD) | MAPE (%) | R² | n_obs |
|---|---|---|---|---|---|
| Naïve (Seasonal) | 28.49 | 31.65 | 78.04 | 0.12 | 274 |
| ARIMA | 14.87 | 17.62 | 35.54 | 0.61 | 274 |
| Standalone LSTM | 5.98 | 7.11 | 18.41 | 0.89 | 258 |
| ARIMA-LSTM Hybrid | 8.43 | 10.22 | 22.17 | 0.84 | 260 |

**Notes on n_obs differences:**
- Naïve and ARIMA use all 274 test observations
- Standalone LSTM: first 30 observations consumed as initial look-back window → 258 predictions
- Hybrid LSTM: first 14 observations consumed as residual look-back window + alignment → 260 predictions

### Key Derived Statistics
- Standalone LSTM MAE improvement over Naïve: **79.0%**
- Standalone LSTM MAE improvement over ARIMA: **59.8%**
- Hybrid MAE improvement over ARIMA: **43.3%**
- Hybrid MAE improvement over Naïve: **70.4%**
- Best model by all metrics: **Standalone LSTM**
- Hybrid is statistically better than ARIMA but NOT statistically better than Standalone LSTM (p = 0.156)

### Walk-Forward CV Results (5-fold, mean MAE)
| Model | CV Mean MAE | CV Std MAE |
|---|---|---|
| Naïve (Seasonal) | 31.24 | ±4.87 |
| ARIMA | 16.43 | ±2.91 |
| Standalone LSTM | 7.82 | ±1.64 |

### SHAP Feature Importance (Hybrid LSTM Exogenous Branch)
Ranked by mean absolute SHAP value:
1. `days_to_departure` — highest importance (booking window effect)
2. `is_high_season` — second highest (demand seasonality)
3. `departure_month_sin` — third (annual cycle)
4. `departure_month_cos` — fourth
5. `departure_day_of_week_sin` — fifth
6. `is_weekend_departure` — sixth
7. `departure_day_of_week_cos` — seventh
8. `departure_quarter` — eighth (lowest but still informative)

---

## PART 8 — METHODOLOGY CHAPTER GUIDE (Chapter 3)

Chapter 3 should cover these sections in order. The code implementing each section is indicated.

### 3.1 Research Design
- Quantitative, longitudinal study
- Secondary data from Kenya Airways booking records
- Comparative model evaluation design
- Positivist epistemology

### 3.2 Data Source and Collection
- Kenya Airways CPV (Customer Profitability Value) booking data
- Route: NBO-MBA, period 2016–2020
- Institutional data access via research agreement
- 5-year period chosen to capture multiple seasonal cycles and structural break (COVID-19)

### 3.3 Data Preprocessing
(See Part 4 of this document)
- Data cleaning, target derivation, feature engineering
- Chronological split rationale: prevents temporal leakage
- Outlier treatment: IQR×3 (conservative — preserves genuine price spikes)

### 3.4 Exploratory Data Analysis
- Descriptive statistics (use tables from Part 3)
- Price series visualisation with rolling statistics
- Seasonal decomposition (STL)
- ACF/PACF analysis for ARIMA order identification
- Stationarity testing (ADF + KPSS)
- Booking window and seasonality analysis

### 3.5 Model Specifications
3.5.1 Naïve baseline  
3.5.2 ARIMA — specify the Zhang (2003) decomposition framework  
3.5.3 Standalone LSTM — specify architecture with equations  
3.5.4 Hybrid ARIMA-LSTM — specify the combination equation: ŷ_t = L_t + N_t where L_t is ARIMA linear component and N_t is LSTM non-linear residual correction  

### 3.6 Hyperparameter Optimisation
- Random search for LSTM hyperparameters
- Grid: units ∈ {32,64,128}, dropout ∈ {0.1,0.2,0.3}, lr ∈ {1e-3,5e-4,1e-4}, batch ∈ {16,32,64}
- Validation MAE as selection criterion
- Final configuration: units=[64,32], dropout=0.20, lr=1e-3 (SA) / 5e-4 (Hybrid)

### 3.7 Model Evaluation Framework
- Metrics: MAE, RMSE, MAPE, R²
- Walk-forward cross-validation (5 folds)
- Diebold-Mariano statistical significance testing
- All evaluation on held-out test set never seen during training or tuning

### 3.8 Uncertainty Quantification
- Monte Carlo Dropout for future forecast confidence intervals
- 100 stochastic forward passes
- 95% CI from empirical distribution of MC samples

### 3.9 Reproducibility
- Random seed = 42 fixed across Python, NumPy, TensorFlow
- `TF_DETERMINISTIC_OPS=1` environment variable
- All artifact versions saved with joblib/keras

### 3.10 Deployment Architecture
(See PART 10 of this document)

### 3.11 Ethical Considerations
- Institutional booking data used with Kenya Airways research agreement
- No personally identifiable information — only aggregated price data
- Commercial sensitivity: raw data not published in thesis, only aggregated statistics
- Algorithmic fairness: forecasting only, no automated pricing decisions without human oversight
- The dashboard includes explicit human oversight guidance (reliability thresholds, analyst review flags)

### 3.12 Limitations
- Single route (NBO-MBA) — results may not generalise to long-haul international routes
- COVID-19 structural break within test set — inflates error metrics
- Exogenous variables limited to booking-derived features — macroeconomic and competitor data not included
- Hybrid model does not outperform Standalone LSTM — suggests residual-correction architecture may require larger dataset for full benefit

---

## PART 9 — DATA ANALYSIS CHAPTER GUIDE (Chapter 4)

Chapter 4 presents the results of EDA and preprocessing. Reference these figures:

### Figures to reference
- **Figure 4.1:** Price series with 30-day rolling mean and standard deviation → `Price_Series_Rolling_Statistics.png`
- **Figure 4.2:** STL seasonal decomposition (trend, seasonal, residual components) → `stl_decomposition.png`
- **Figure 4.3:** ACF and PACF plots for ARIMA order selection → `acf_pacf.png`
- **Figure 4.4:** Training/Validation/Test split timeline

### Key findings to narrate
1. The price series exhibits clear weekly seasonality (period = 7) and an upward trend from 2016–2019
2. COVID-19 caused a structural break in March 2020 — prices collapsed then became highly volatile
3. The series is non-stationary in levels (ADF p > 0.05) but stationary after first differencing (ADF p < 0.01), confirming I(1) process
4. KPSS test corroborates: fails to reject stationarity of differenced series
5. ACF shows significant spikes at lags 7, 14, 21 (weekly seasonality) — informs ARIMA seasonal component
6. PACF cuts off quickly after lag 2 — informs AR order
7. Last-minute bookings carry a 33.8% premium over very long advance bookings
8. High-season prices are 38.2% higher than low-season

### Booking window analysis table
Use exact values from Part 3, Section "Booking Window Distribution".

### Stationarity test table
| Test | Statistic | p-value | Conclusion |
|---|---|---|---|
| ADF (levels) | report from run | > 0.05 | Non-stationary |
| ADF (1st diff) | report from run | < 0.01 | Stationary |
| KPSS (levels) | report from run | < 0.05 | Non-stationary |
| KPSS (1st diff) | report from run | > 0.05 | Stationary |

---

## PART 10 — MODEL EVALUATION CHAPTER GUIDE (Chapter 5)

### Structure
5.1 Baseline: Naïve Seasonal Model  
5.2 ARIMA Model — specification, diagnostics, results  
5.3 Standalone LSTM — architecture, training curves, results  
5.4 ARIMA-LSTM Hybrid — architecture, residual analysis, results  
5.5 Walk-Forward Cross-Validation  
5.6 Statistical Significance (Diebold-Mariano)  
5.7 Comprehensive Comparison  
5.8 SHAP Feature Importance  
5.9 Future Forecast (30-day ahead)  

### Figures to reference
- **Figure 5.1:** `Standalone_LSTM_training.png` — loss and MAE training curves
- **Figure 5.2:** `ARIMA-LSTM_Hybrid_training.png` — loss and MAE training curves
- **Figure 5.3:** `three_way_comparison.png` — full timeline + per-model test panels
- **Figure 5.4:** `metric_comparison.png` — bar charts MAE/RMSE/MAPE/R²
- **Figure 5.5:** `cv_mae_comparison.png` — walk-forward CV box plots
- **Figure 5.6:** `shap_feature_importance.png` — SHAP bar chart
- **Figure 5.7:** `future_forecast_all_models.png` — 30-day forecast with MC CI bands

### Results tables to include
Use exact numbers from Part 7 of this document.

**Table 5.1:** Test-set performance comparison (MAE, RMSE, MAPE, R²)  
**Table 5.2:** Walk-forward CV results (mean ± std MAE)  
**Table 5.3:** Diebold-Mariano test results  
**Table 5.4:** Model rank by metric  

---

## PART 11 — DISCUSSION CHAPTER GUIDE (Chapter 6)

### Key arguments to make

**6.1 Standalone LSTM dominance**
The Standalone LSTM achieving MAE = 5.98 USD (MAPE = 18.41%) is consistent with the literature finding that LSTM networks outperform classical statistical models for non-linear price series with complex seasonal patterns (Hochreiter & Schmidhuber 1997, Fischer & Krauss 2018). The 79% MAE improvement over the naïve baseline and 59.8% improvement over ARIMA demonstrate that non-linear patterns exist in the NBO-MBA price series that ARIMA cannot capture.

**6.2 Why the Hybrid did not outperform Standalone LSTM**
The hybrid model (MAE = 8.43) is significantly better than ARIMA (p = 0.005) but not significantly different from the Standalone LSTM (p = 0.156). This is an important finding that requires discussion. Two explanations:

First, the residual correction architecture assumes ARIMA captures the linear signal completely, leaving only non-linear patterns for the LSTM to learn. If ARIMA's linear representation is imperfect (as is likely given the complex seasonality and COVID-19 disruption), the residuals contain both non-linear signal AND noise from ARIMA's misspecification. The LSTM then learns partly corrupted targets.

Second, the training dataset for the hybrid LSTM is small (~1,000 sequences of length 14). The standalone LSTM trains on ~1,000 sequences of length 30 on the raw price signal — a richer, more informative target. With a larger dataset (multiple routes, multiple years), the hybrid architecture would likely show greater benefit.

This is consistent with Zhang (2003)'s original hybrid framework, which was evaluated on longer series (airline passenger data, 144 months) where the hybrid more clearly outperformed components.

**6.3 Practical significance**
A MAPE of 18.41% means the Standalone LSTM forecasts are on average 18.41% away from actual prices. For a USD 350 average ticket, this is approximately USD 64. This may be acceptable for revenue management trend analysis (is demand rising or falling?) but insufficient for automated precision pricing. Human oversight remains essential — which is why the dashboard includes explicit reliability thresholds and analyst review flags.

**6.4 Booking window as primary determinant**
SHAP analysis confirms that `days_to_departure` is the most important feature for the hybrid model's residual correction — validating SO1. Last-minute pricing is the strongest determinant of price variability, consistent with airline revenue management theory (Talluri & van Ryzin 2004).

**6.5 COVID-19 confound**
The test set (approximately October 2019 – December 2020) overlaps with the COVID-19 disruption. This means reported error metrics are likely inflated compared to a stable operating period. Future work should evaluate models on pre-COVID and post-COVID periods separately.

**6.6 Comparison to literature**
Compare results to:
- Tziridis et al. (2017): MAPE of 18–25% for airline price forecasting with ARIMA
- Pang et al. (2017): LSTM MAPE of 6–12% for flight prices (longer time series, international routes)
- Cao & Tay (2001): hybrid ARIMA-SVM outperforming individual models on S&P 500
- Zhang (2003): hybrid ARIMA-ANN improvements of 20–40% over ARIMA on airline passenger data

---

## PART 12 — CONCLUSIONS CHAPTER GUIDE (Chapter 7)

### 7.1 Summary of Findings (answer each research question directly)

**SRQ1:** Advance purchase window (`days_to_departure`) is the primary price determinant (SHAP rank 1), followed by high-season indicator and monthly seasonality. Last-minute bookings carry a 33.8% premium; high-season fares are 38.2% higher than low-season.

**SRQ2:** ARIMA achieves MAE = 14.87 (MAPE = 35.54%), significantly outperforming the naïve baseline (p = 0.002). Standalone LSTM achieves MAE = 5.98 (MAPE = 18.41%), significantly outperforming both the naïve baseline and ARIMA (p < 0.001).

**SRQ3:** The hybrid ARIMA-LSTM model achieves MAE = 8.43 (MAPE = 22.17%), significantly better than ARIMA (p = 0.005) but not significantly different from the Standalone LSTM (p = 0.156). Combining ARIMA and LSTM improves over ARIMA alone but does not improve over the Standalone LSTM on this dataset.

**SRQ4:** All learned models are statistically significantly better than the naïve baseline. The Standalone LSTM is statistically significantly better than ARIMA. The hybrid-vs-LSTM difference is not statistically significant, suggesting that on this route and dataset size, the added complexity of the hybrid architecture does not deliver measurable additional benefit.

### 7.2 Theoretical Contributions
- Demonstrates applicability of the Zhang (2003) hybrid ARIMA-ANN framework to sub-Saharan African airline pricing data
- Identifies advance purchase window as the dominant non-linear price driver in Kenya Airways NBO-MBA pricing, extending findings from European and North American airline pricing studies
- Provides empirical evidence that the hybrid architecture's benefit may be dataset-size dependent

### 7.3 Practical Contributions
- A deployed forecasting system (dashboard + API) suitable for integration into Kenya Airways revenue management workflows
- A documented pipeline for replicating and updating the analysis as new data becomes available
- Booking window and seasonal pricing insight tables directly applicable to yield management strategy

### 7.4 Recommendations
1. **Deploy Standalone LSTM** as the primary forecasting model for NBO-MBA revenue management support
2. **Expand to additional routes** — NBO-LHR, NBO-DXB — where longer series may allow the hybrid to demonstrate its full benefit
3. **Incorporate competitor pricing data** as additional exogenous features for the hybrid model
4. **Re-evaluate post-COVID** on clean 2022–2024 data once the COVID-19 structural break is outside the evaluation window
5. **Extend hybrid CV** — implement walk-forward CV for the hybrid model despite computational cost, possibly using cloud GPU

### 7.5 Limitations
1. Single route analysis limits generalisability
2. COVID-19 structural break confounds test-set metrics
3. No competitor or macroeconomic variables
4. Hybrid CV not implemented (computational constraint)

### 7.6 Areas for Further Research
1. Multi-route generalisation study
2. Transformer / attention-based architectures for airline pricing
3. Online learning — model updates as new data arrives without full retraining
4. Demand forecasting integration — combining price and passenger volume models
5. Interpretability of the Standalone LSTM (currently a black box beyond SHAP)

---

## PART 13 — DEPLOYMENT ARCHITECTURE (for Methodology and Conclusions)

The thesis describes a four-component deployment architecture:

### Component 1: Training Pipeline (`air_ticket_price_forecasting_thesis.py`)
- Runs once to train all three models
- Produces 8 artifact files in `/artifacts/`
- Generates all evaluation plots and the future forecast CSV

### Component 2: Production Inference Pipeline (`pipeline.py`)
- Runs daily (scheduled via cron or Windows Task Scheduler)
- Loads trained artifacts from `/artifacts/`
- Accepts new booking data via command line: `python pipeline.py --data bookings.xlsx`
- Writes forecasts, actuals, performance metrics, and run audit records to `forecasting.db`

### Component 3: REST API (`api.py`)
- FastAPI service, port 8000
- **Sole gateway** between the database and all consumers — the dashboard and any external system communicate only through the API, never directly with the SQLite file
- 8 endpoints (see README for full list)
- Auto-generated interactive documentation at `http://localhost:8000/docs` (OpenAPI/Swagger)
- Model artifacts loaded once at startup (not per-request) for sub-50ms response times

### Component 4: Streamlit Dashboard (`dashboard.py`)
- Interactive web UI for revenue management analysts
- Calls the REST API exclusively — contains no direct database code
- 6 tabs: Forecasts, Test-Set Results, Forecast Analysis, Booking Intelligence, Custom Forecast, Pipeline Status
- Custom Forecast tab: analysts can paste their own price history and generate on-demand forecasts from all three models without re-running the pipeline

### Architectural Principle
The API-first design follows the separation of concerns principle. Authentication, rate limiting, input validation, and logging are defined once in the API and apply uniformly to all consumers. The dashboard and any future external system (Excel plugin, mobile pricing app, revenue management integration) consume identical endpoints, ensuring consistency.

In production at Kenya Airways, the API would be:
- Containerised with Docker
- Deployed to a cloud VM or Kubernetes pod
- Secured with API-key or OAuth2 authentication
- Monitored via the `/health` endpoint queried by an Airflow DAG

---

## PART 14 — FILES REQUIRED FOR THESIS GENERATION

Upload ALL of the following files when asking an LLM to generate thesis chapters:

### Essential (must have)
| File | Why needed |
|---|---|
| `PROJECT_SUMMARY.md` | This file — all numbers, rationale, structure |
| `README.md` | Project overview and run instructions |
| `air_ticket_price_forecasting_thesis.py` | Complete training code with inline THESIS NOTEs explaining every architectural decision |
| `pipeline.py` | Production pipeline for deployment chapter |
| `api.py` | REST API for deployment chapter |
| `dashboard.py` | Dashboard for deployment chapter |

### Strongly recommended (include if available)
| File | Why needed |
|---|---|
| `chapter3_methodology.tex` | Already-written Chapter 3 — do not regenerate, use as reference and extend |
| `Chapter3_Methodology.docx` | Word version of Chapter 3 — same content |
| `future_forecast_all_models.csv` | Actual forecast values for results tables |

### Include if available (enrich results)
| File | Why needed |
|---|---|
| `Price_Series_Rolling_Statistics.png` | Figure for Chapter 4 |
| `stl_decomposition.png` | Figure for Chapter 4 |
| `acf_pacf.png` | Figure for Chapter 4 |
| `three_way_comparison.png` | Figure for Chapter 5 |
| `metric_comparison.png` | Figure for Chapter 5 |
| `cv_mae_comparison.png` | Figure for Chapter 5 |
| `shap_feature_importance.png` | Figure for Chapter 5/6 |
| `future_forecast_all_models.png` | Figure for Chapter 5/7 |
| `Standalone_LSTM_training.png` | Figure for Chapter 5 |
| `ARIMA-LSTM_Hybrid_training.png` | Figure for Chapter 5 |

### Do not upload
| File | Why not needed |
|---|---|
| `CPV Data NBO-MBA 2016-2020.xlsx` | Commercially sensitive raw data — use summary statistics from this document |
| `forecasting.db` | Binary SQLite file — not readable |
| `/artifacts/*.pkl` / `*.keras` | Binary model files — not readable |

---

## PART 15 — CITATION KEYS (for LaTeX / Word chapters)

These citation keys match the `.bib` file from the original proposal. Use them in `\citep{}` / `\citet{}` commands:

| Reference | Key |
|---|---|
| Zhang (2003) hybrid ARIMA-ANN | `zhang2003` |
| Hochreiter & Schmidhuber (1997) LSTM | `hochreiter1997` |
| Box & Jenkins (1970) ARIMA | `box1970` |
| Diebold & Mariano (1995) DM test | `diebold1995` |
| Harvey, Leybourne & Newbold (1997) DM correction | `harvey1997` |
| Gal & Ghahramani (2016) MC Dropout | `gal2016` |
| Khashei & Bijari (2011) hybrid review | `khashei2011` |
| Fischer & Krauss (2018) LSTM finance | `fischer2018` |
| Talluri & van Ryzin (2004) revenue management | `talluri2004` |
| Tziridis et al. (2017) airline ARIMA | `tziridis2017` |
| Pang et al. (2017) airline LSTM | `pang2017` |
| Cao & Tay (2001) hybrid SVM | `cao2001` |

---

## PART 16 — WRITING STYLE NOTES

When generating chapters, follow these conventions:

1. **Academic register** — formal third person, no contractions
2. **Present tense for describing the methodology**, past tense for reporting results
3. **All equations** should be numbered and referenced in text
4. **All figures** should be numbered, captioned, and explicitly referenced in text: "as shown in Figure 5.3"
5. **All tables** should be numbered and have descriptive captions
6. **Statistical results** should state: statistic value, degrees of freedom (where applicable), and p-value
7. **Never invent numbers** — use only values in this document or values that will be computed when the code is run
8. **Placeholders** — where a value is runtime-dependent (e.g. exact ARIMA order, exact stationarity test statistics), use `[INSERT FROM RUN OUTPUT]` as a placeholder
9. **Chapter 3 already exists** in `chapter3_methodology.tex` — do not regenerate it from scratch. If Chapter 3 is needed, extend or adapt the existing document
10. **Word count targets:** Chapter 3 ~3,500 words, Chapter 4 ~2,500 words, Chapter 5 ~4,000 words, Chapter 6 ~3,000 words, Chapter 7 ~1,500 words
