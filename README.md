# HITACHI_Case_Study

# 💰 ATM Cash Forecasting using XGBoost

This project addresses a real-world operations challenge for Hitachi's cash business in India, where more than 20,000 ATMs are managed for multiple banks. The goal is to **forecast daily cash dispense amounts for each ATM for the next 7 days** in order to prevent penalties due to:

- 💸 **Idle cash** (overloading ATMs that dispense less)
- ❌ **Cash outs** (underloading ATMs that run out of money)

---

## 📦 Dataset Overview

The dataset includes **2 years of daily ATM-level data** across 3 banks, with the following features:

| Column          | Description |
|----------------|-------------|
| `Bank`         | Bank to which ATM belongs |
| `ATMID`        | Unique identifier for ATM |
| `Caldate`      | Date of record |
| `Dispense`     | Amount dispensed on that day |
| `DT`           | Downtime in minutes |
| `MaxCapacity`  | Max cash capacity of ATM |
| `CountTotalTxn`| Number of transactions |

---

## ⚙️ Methodology

1. **Data Preparation**  
   Cleaned and converted dates, sorted, and stripped column names. Changed Account to Bank and ATMID' to ATMID fro original data.raw.xlsx

2. **Feature Engineering**  
   - Day of week and weekend flag  
   - Downtime ratio  
   - Lag features (1-day, 7-day)  
   - 7-day rolling average  
   - Utilization %

3. **Model Training**  
   - Trained one **XGBoost Regressor per bank**  
   - Used recent historical data to build per-ATM time series features  
   - Evaluated using **Mean Absolute Error (MAE)**

4. **Forecasting Logic**  
   - Forecasted **next 7 days** per ATM  
   - Ensured dispense prediction ≤ MaxCapacity  
   - Assumed no downtime for forecast window
   

5. **Outputs**  
   - CSV with forecasts: `atm_forecast_7days.csv`  

---

## 📊 Sample Forecast Output

| Bank | ATMID | Date       | Predicted_Dispense |
|------|-------|------------|--------------------|
| B1   | A101  | 2025-04-01 | ₹145,000           |
| B1   | A101  | 2025-04-02 | ₹138,500           |
| ...  | ...   | ...        | ...                |

---

## 🛠️ How to Run

### Requirements
Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn python-pptx openpyxl

