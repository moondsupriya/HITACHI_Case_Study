# Imports and Initial Setup
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load the Data
data_path = 'data_raw.xlsx'  
df = pd.read_excel(data_path)
df.head()

#  Basic Info and Data Types
print("\nData Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe()) 

#prepare the Data
# Debugging: Print column names
print("Columns in the dataset:", df.columns)

# Strip spaces from column names (if any)
df.columns = df.columns.str.strip()

# Ensure the column exists
if 'Caldate' not in df.columns:
    raise KeyError("The column 'Caldate' is missing from the dataset. Please check the file.")

# Convert 'Caldate' to datetime
df['Caldate'] = pd.to_datetime(df['Caldate'])
df.sort_values(['Bank', 'ATMID', 'Caldate'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Feature Engineering
def create_features(df):
    df['DayOfWeek'] = df['Caldate'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
    df['DT_ratio'] = df['DT'] / (24*60)
    df['Utilization'] = df['Dispense'] / df['MaxCapacity']
    
    # Lag and rolling features
    df['Lag_1'] = df.groupby('ATMID')['Dispense'].shift(1)
    df['Lag_7'] = df.groupby('ATMID')['Dispense'].shift(7)
    df['RollingMean_7'] = df.groupby('ATMID')['Dispense'].shift(1).rolling(7).mean().reset_index(0, drop=True)
    
    return df

df = create_features(df)

# Train XGBoost Models Per Bank
df_model = df.dropna()
features = ['DayOfWeek', 'IsWeekend', 'DT_ratio', 'CountTotalTxn', 'Lag_1', 'Lag_7', 'RollingMean_7']
target = 'Dispense'
bank_models = {}

for bank in df_model['Bank'].unique():
    df_bank = df_model[df_model['Bank'] == bank]
    X = df_bank[features]
    y = df_bank[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"[{bank}] MAE: {mean_absolute_error(y_test, y_pred):,.0f}")
    
    bank_models[bank] = model


#  Forecast Next 7 Days
forecast_days = 7
future_preds = []
last_date = df['Caldate'].max()

for bank in df['Bank'].unique():
    model = bank_models[bank]
    
    for atmid in df[df['Bank'] == bank]['ATMID'].unique():
        df_atm = df[(df['Bank'] == bank) & (df['ATMID'] == atmid)].copy()
        
    for i in range(forecast_days):
        last_row = df_atm.iloc[-1:].copy()
        pred_date = last_row['Caldate'].values[0] + np.timedelta64(1, 'D')
        
        # Convert pred_date to pandas.Timestamp to use .weekday()
        pred_date = pd.Timestamp(pred_date)
        
        new_row = {
            'Caldate': pred_date,
            'Bank': bank,
            'ATMID': atmid,
            'DayOfWeek': pred_date.weekday(),
            'IsWeekend': int(pred_date.weekday() in [5, 6]),
            'DT_ratio': 0,  # Assume no downtime
            'CountTotalTxn': last_row['CountTotalTxn'].values[0],
            'Lag_1': last_row['Dispense'].values[0],
            'Lag_7': df_atm.iloc[-7]['Dispense'] if len(df_atm) >= 7 else np.nan,
            'RollingMean_7': df_atm['Dispense'].tail(7).mean(),
            'MaxCapacity': last_row['MaxCapacity'].values[0]
        }
        
        X_pred = pd.DataFrame([new_row])[features]
        y_pred = model.predict(X_pred)[0]
        y_pred_clipped = min(y_pred, new_row['MaxCapacity'])  # Clip to max
        
        new_row['Dispense'] = y_pred_clipped
        df_atm = pd.concat([df_atm, pd.DataFrame([new_row])], ignore_index=True)
        
        future_preds.append({
            'Bank': bank,
            'ATMID': atmid,
            'Date': pred_date,
            'Predicted_Dispense': y_pred_clipped
        })

forecast_df = pd.DataFrame(future_preds)

#  Export or View Forecast
forecast_df.to_csv('atm_forecast_7days.csv', index=False)
forecast_df.head(10)


