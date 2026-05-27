# 🛒 Retail Sales Forecasting System

End-to-end time series forecasting pipeline comparing SARIMA, Prophet, 
and XGBoost on 2 years of daily retail sales data.

## Results
| Model | MAPE | MAE | RMSE |
|---|---|---|---|
| SARIMA | 18.91% | $357 | $569 |
| Prophet | 15.09% | $229 | $396 |
| **XGBoost** | **12.00%** | **$185** | **$285** |

## Tech Stack
Python, Pandas, NumPy, Statsmodels, Prophet, XGBoost, Matplotlib, Seaborn, Scikit-learn

## Project Steps
1. Generated 730 days of realistic retail sales data (trend, seasonality, holidays, promotions)
2. EDA — distributions, correlations, boxplots, rolling averages
3. ADF stationarity testing — differenced series to achieve p=0.000
4. ACF/PACF analysis — confirmed weekly seasonality at lag 7, 14, 21
5. Seasonal decomposition — separated trend, seasonality, residuals
6. SARIMA(1,1,1)(1,1,1)[7] model — 18.91% MAPE
7. Prophet with custom holiday events — 15.09% MAPE
8. XGBoost with 15 lag and rolling window features — 12.00% MAPE
9. Automated model comparison dashboard

## Key Findings
- XGBoost outperformed statistical models by 37% (MAPE)
- lag_1 (yesterday's sales) was the strongest predictor (importance=0.28)
- Holiday spikes are the hardest pattern to forecast
- Prophet handled holidays better than SARIMA due to explicit event modeling

## Files
- `forecasting.ipynb` — main notebook with all steps and charts
- `retail_data.csv` — generated retail sales dataset
