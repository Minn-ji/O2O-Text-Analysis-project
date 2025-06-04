import matplotlib.pyplot as plt
import pandas as pd

# 누적 리뷰 수 추이

def plot_cumulative_reviews(df):
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    daily = df['date'].dt.date.value_counts().sort_index()
    cumulative = daily.cumsum()

    plt.figure(figsize=(14, 5))
    plt.plot(cumulative.index, cumulative.values)
    plt.title('누적 리뷰 수 추이')
    plt.xlabel('날짜')
    plt.ylabel('누적 리뷰 수')
    plt.grid(alpha=0.3)
    plt.tight_layout()