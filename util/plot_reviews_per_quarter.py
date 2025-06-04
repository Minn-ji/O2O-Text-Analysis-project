import matplotlib.pyplot as plt
import pandas as pd

# 분기 별 리뷰 수 그래프 작성

def plot_reviews_per_quarter(df):
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    df['quarter'] = df['date'].dt.to_period('Q')
    quarter_counts = df['quarter'].value_counts().sort_index()

    plt.figure(figsize=(12, 5))
    plt.bar(quarter_counts.index.astype(str), quarter_counts.values)
    plt.title('분기별 리뷰 개수')
    plt.xlabel('분기')
    plt.ylabel('리뷰 개수')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()