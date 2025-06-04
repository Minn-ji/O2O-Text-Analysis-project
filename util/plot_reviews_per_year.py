import matplotlib.pyplot as plt
import pandas as pd

# 연도 별 리뷰 수

def plot_reviews_per_year(df):
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    year_counts = df['date'].dt.year.value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    plt.bar(year_counts.index.astype(str), year_counts.values)
    plt.title('연도별 리뷰 개수')
    plt.xlabel('연도')
    plt.ylabel('리뷰 개수')
    plt.grid(axis='y', alpha=0.3)
    plt.show()