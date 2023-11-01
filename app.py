import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/daniloui001/Correlation/main/movies.csv')
pd.set_option('display.max_columns', None)

# Overview

head = df.head()
print(head)

print(df.columns)

# Identifying Nulls

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {} %'.format(col, round(pct_missing*100)))

# Rating 1% nulls, Gross 2% nulls, Budget 28% nulls

# Dropping nulls

df = df.dropna()

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {} %'.format(col, round(pct_missing*100)))

# Data Cleaning

data_types = df.dtypes
print(data_types)

df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')

df['yearclarified'] = df['released'].astype(str).str[:4]

# Identifying Duplicates and dropping them

duplicate_company = df['company'].drop_duplicates().sort_values(ascending=False)
print(duplicate_company)

df.drop_duplicates()