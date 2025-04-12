import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("C:/Users/Lenovo/Downloads/Video+Game+Sales (1)/vgchartz-2024.csv")


df.drop(columns=['img', 'last_update'], inplace=True)

# Fill missing developer names with 'Unknown'
df['developer'] = df['developer'].fillna('Unknown')

# Fill missing critic scores with the mean
df['critic_score'] = df['critic_score'].fillna(df['critic_score'].mean())

# Fill missing sales values with 0
sales_columns = ['total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales']
df[sales_columns] = df[sales_columns].fillna(0)

# Convert release_date to datetime format
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Drop rows with invalid dates
df.dropna(subset=['release_date'], inplace=True)

# Extract release year
df['release_year'] = df['release_date'].dt.year

# Recalculate total_sales if missing (but regional data is present)
regional_sales = df[['na_sales', 'jp_sales', 'pal_sales', 'other_sales']].sum(axis=1)
df['total_sales'] = df.apply(
    lambda row: regional_sales[row.name] if row['total_sales'] == 0 else row['total_sales'], axis=1)

df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)


# Display cleaned data summary
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


