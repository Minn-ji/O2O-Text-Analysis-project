import matplotlib.pyplot as plt
import pandas as pd

# 일 별 리뷰 수 그래프 작성

def plot_reviews_for_month(df, year, month):
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    mask = (df['date'].dt.year == year) & (df['date'].dt.month == month)
    monthly_df = df.loc[mask].copy()

    if monthly_df.empty:
        print(f"{year}-{month:02d}에는 리뷰가 없습니다.")
        return

    daily_counts = monthly_df['date'].dt.day.value_counts().sort_index()

    plt.figure(figsize=(10, 4))
    plt.bar(daily_counts.index, daily_counts.values)
    plt.title(f'{year}년 {month:02d}월 일별 리뷰 개수')
    plt.xlabel('일')
    plt.ylabel('리뷰 개수')
    plt.xticks(daily_counts.index)
    plt.grid(axis='y', alpha=0.3)
    plt.show()