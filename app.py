import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from collections import Counter

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

pattern = r'(\d{4})'

df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')

df['score'] = df['score'].astype('float64')
df['year'] = df['year'].astype('int64')

df['yearclarified'] = df['released'].astype(str).str.extract(pattern)
df['yearclarified'] = df['yearclarified'].astype('int64')

df = df.sort_values(by=['yearclarified'], ascending=False)

# Identifying Duplicates and dropping them

duplicate_company = df['company'].drop_duplicates().sort_values(ascending=False)
print(duplicate_company)

df.drop_duplicates()

# Scatterplot: budget vs. gross revenue

plt.scatter(x=df['budget'], y=df['gross'])
plt.title("Budget vs. Gross Earnings")
plt.xlabel("Budget")
plt.ylabel("Gross Earnings")
plt.show()

# Plot the budget vs. gross using seaborn

correlation_matrix = df.corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation for Numeric Features")
plt.xlabel("Movie Features")
plt.ylabel("Movie Features")
plt.show()

sns.regplot(x='budget',y='gross',data=df,scatter_kws={"color": "red"}, line_kws={"color": "blue"})
plt.show()

# Company

df_numerical = df

for col_name in df_numerical.columns:
    if df_numerical[col_name].dtype == 'object':
        df_numerical[col_name] = df_numerical[col_name].astype('category')
        df_numerical[col_name] = df_numerical[col_name].cat.codes


print(df_numerical)

correlation_matrix = df_numerical.corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation for Numeric Features")
plt.xlabel("Movie Features")
plt.ylabel("Movie Features")
plt.show()

df_numerical.corr()

# Utilizing unstacking

correlation_matrix = df_numerical.corr()

corr_paris = correlation_matrix.unstack()
print(corr_paris)

sorted_pairs = corr_paris.sort_values()

high_correlation = sorted_pairs[(sorted_pairs) > 0.5]

# Correlation between score of movies over time based on genre

unique_genres = set(df['genre'])
print("Unique Genres: ")
for genre in unique_genres:
    print(genre)
num_unique_genres = len(unique_genres)

print(num_unique_genres)

# 15 unique genres

yearly_scores = df.groupby('yearclarified')['score'].mean()
df

for genre in unique_genres:
    genre_data = df[df['genre'] == genre]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(genre_data['yearclarified'], genre_data['score'])
    
    # Set labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Score')
    ax.set_title(f'{genre} Scores Over Time')

    ax.set_xlim(1980, 2020)
    ax.set_ylim(0, 10)
    
    plt.show()

# Correlation between score of movies over time based on genre

for genre in unique_genres:
    genre_data = df[df['genre'] == genre]

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=genre_data['yearclarified'], y=genre_data['score'], hue='genre', markers=True, markersize=8)
    plt.title('Movie Scores by Genre Over Time')
    plt.xlabel('Year')
    plt.ylabel('Score')
    plt.legend(title='Genre', loc='best')
    plt.show()