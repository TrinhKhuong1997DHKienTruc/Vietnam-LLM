# S&P 500 Forecast Report – Next 5 Trading Days

*Generated: 2025-08-16*

---

## 1. Overview
This report presents a short-term forecast of the S&P 500 (^GSPC) closing price for the **next five trading days** using a simple **ARIMA(5, 1, 0)** time–series model trained on the last **3 years** of daily data (≈ 753 observations, 2022-08-16 → 2025-08-15).

> **Important** This is a demo forecast intended to illustrate the workflow. It is **not** financial advice. Accuracy may be limited by the model’s simplicity and absence of exogenous variables.

---

## 2. Data Summary
| Item | Value |
|------|-------|
| Ticker | ^GSPC (S&P 500) |
| Source | Yahoo Finance (`yfinance`) |
| Date range | 2022-08-16 → 2025-08-15 |
| Observations | 753 daily closes |

---

## 3. Model Specification
* **Model type**: ARIMA (Autoregressive Integrated Moving-Average)
* **Order**: (p = 5, d = 1, q = 0)
* **Software**: `statsmodels` 0.14

### 3.1 Diagnostic Statistics
```
Dep. Variable: Close  |  Observations: 753
Model: ARIMA(5,1,0)   |  Log-Likelihood: ‑4028.99
AIC: 8069.97          |  BIC: 8097.71
sigma² (resid var): 2637.71

Coefficients:
  ar1  -0.0331  (p=0.101)
  ar2   0.0395  (p=0.246)
  ar3  -0.0912  (p=0.001) **
  ar4  -0.0734  (p=0.002) **
  ar5   0.0193  (p=0.435)

Residual diagnostics:
  Ljung-Box p-value (lag 1) = 0.91 ➜ no autocorrelation
  Heteroskedasticity H = 1.97 (p ≈ 0.00) ➜ evidence of non-constant variance
  Jarque-Bera = 2873.6 (p ≈ 0.00) ➜ residuals are non-normal
```

*`**` indicates statistical significance at 1 % level.*

Interpretation: The model fits reasonably (AIC ≈ 8070) but residual diagnostics show heteroskedasticity and non-normality, common in financial time-series. Forecasts should therefore be interpreted cautiously.

---

## 4. Forecast (Point Estimates)
| Date | Forecast Close |
|------|---------------:|
| 2025-08-18 | **6 442.98** |
| 2025-08-19 | **6 442.15** |
| 2025-08-20 | **6 443.88** |
| 2025-08-21 | **6 445.82** |
| 2025-08-22 | **6 446.04** |

*(Values in USD; weekends skipped.)*

### 4.1 Trend Note
The point estimates suggest a **modest upward drift (~ 0.05 % total)** over the forecast horizon, consistent with recent positive momentum in the historical series.

---

## 5. Limitations & Next Steps
1. **Model Simplicity** – ARIMA ignores volatility clustering and exogenous macro variables. Consider GARCH or machine-learning models for better accuracy.
2. **Parameter Stability** – Coefficients may vary if re-estimated with new data. Re-train regularly.
3. **Uncertainty Bands** – Only point forecasts are shown. Add confidence intervals for risk assessment.
4. **Weekend/holiday handling** – Ensure calendar alignment when applying forecasts.

---

## 6. Reproducibility
Run the forecast locally:
```bash
source Deep-Reinforcement-Stock-Trading/venv/bin/activate
python forecast_sp500_nextweek.py
```
The script fetches fresh data, fits the model, and prints an updated forecast.

---

*Report generated automatically by `forecast_sp500_nextweek.py`.*