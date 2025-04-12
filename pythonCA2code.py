import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("C:/Users/Lenovo/Downloads/Video+Game+Sales (1)/vgchartz-2024.csv")

df.drop(columns=['img', 'last_update'], inplace=True)
df['developer'] = df['developer'].fillna('Unknown')
df['critic_score'] = df['critic_score'].fillna(df['critic_score'].mean())

sales_columns = ['total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales']
df[sales_columns] = df[sales_columns].fillna(0)

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df.dropna(subset=['release_date'], inplace=True)
df['release_year'] = df['release_date'].dt.year

regional_sales = df[['na_sales', 'jp_sales', 'pal_sales', 'other_sales']].sum(axis=1)
df['total_sales'] = df.apply(
    lambda row: regional_sales[row.name] if row['total_sales'] == 0 else row['total_sales'], axis=1
)

df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print(df.info())
print(df.head())

# Bar Plot – Total Sales by Console
plt.figure(figsize=(14, 6))
sns.barplot(data=df, x='console', y='total_sales', estimator='sum', ci=None, palette='viridis')
plt.xticks(rotation=90)
plt.title("Total Sales by Console")
plt.tight_layout()
plt.show()

# Count Plot – Top 10 Genres
top_genres = df['genre'].value_counts().nlargest(10).index
plt.figure(figsize=(10, 6))
sns.countplot(data=df[df['genre'].isin(top_genres)], x='genre', order=top_genres, palette='Set2')
plt.xticks(rotation=45)
plt.title("Top 10 Genres by Game Count")
plt.tight_layout()
plt.show()

# Line Plot – Total Sales Over Years
sales_by_year = df.groupby('release_year')['total_sales'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=sales_by_year, x='release_year', y='total_sales', marker='o')
plt.title("Total Sales Over the Years")
plt.xlabel("release_year")
plt.ylabel("Total Sales (millions)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Heatmap – Correlation Between Numeric Columns
plt.figure(figsize=(8, 6))
sns.heatmap(df[['critic_score', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales', 'total_sales']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Numeric Columns")
plt.tight_layout()
plt.show()

# Box Plot – Critic Score Distribution by Genre (Top 10 only)
plt.figure(figsize=(14, 6))
sns.boxplot(data=df[df['genre'].isin(top_genres)], x='genre', y='critic_score', palette='pastel')
plt.xticks(rotation=45)
plt.title("Critic Score by Genre")
plt.tight_layout()
plt.show()

# Scatter Plot – Critic Score vs Total Sales
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='critic_score', y='total_sales', hue='genre', legend=False, alpha=0.6)
plt.title("Critic Score vs Total Sales")
plt.xlabel("Critic Score")
plt.ylabel("Total Sales (millions)")
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Encode categorical columns
df_model = df.copy()
le_console = LabelEncoder()
le_genre = LabelEncoder()
le_publisher = LabelEncoder()

df_model['console'] = le_console.fit_transform(df_model['console'])
df_model['genre'] = le_genre.fit_transform(df_model['genre'])
df_model['publisher'] = le_publisher.fit_transform(df_model['publisher'])

# Define features and target (use log of sales to reduce skew)
X = df_model[['console', 'genre', 'publisher', 'critic_score', 'release_year']]
y = np.log1p(df_model['total_sales'])  # log(1 + x) transformation

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and inverse-transform the log back to normal scale
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # inverse of log1p
y_actual = np.expm1(y_test)

# Evaluation
mse = mean_squared_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)
print("Mean Squared Error:", round(mse, 2))
print("R^2 Score:", round(r2, 2))

# 📈 Plot: Actual vs Predicted Total Sales 
plt.figure(figsize=(8, 6))
plt.scatter(y_actual, y_pred, alpha=0.5, color='green')
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Total Sales")
plt.ylabel("Predicted Total Sales")
plt.title("Improved: Actual vs Predicted Total Sales (Random Forest)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
