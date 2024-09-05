##### Task 2 ####

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('C:/Users/91960/Desktop/python_ws/test.csv')
print(df)
print(df.isnull().sum())
df.dropna(subset=["Embarked"],inplace=True)
df["Cabin"]=df["Cabin"].fillna("Unknown")
df["Age"]=df["Age"].fillna(df["Age"].mean())
print(df.isnull().sum())
print(df.duplicated().sum())
from scipy import stats

z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
df_clean = df[(z_scores < 3).all(axis=1)]

print(df.describe())
print(df.info())

# Histograms for numerical features
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Boxplots for detecting outliers
for column in df.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(y=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()


# Select only numeric columns
numeric_df = df.select_dtypes(include=[float, int])



# Compute the correlation matrix
corr_matrix = numeric_df.corr()

# Print the correlation matrix
print("Correlation Matrix:")
print(corr_matrix)

# Plot heatmap of the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Scatter plot between two variables
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', data=df)
plt.title('Scatter Plot between variable1 and variable2')
plt.show()

# Pair plot for a subset of features
sns.pairplot(df[['Sex', 'Age', 'Fare']])
plt.show()



