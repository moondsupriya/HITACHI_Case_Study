# Imports and Initial Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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