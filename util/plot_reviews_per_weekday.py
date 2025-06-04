import matplotlib.pyplot as plt
import pandas as pd

# 요일별 리뷰 빈도

def plot_reviews_per_weekday(df):
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    df['weekday'] = df['date'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = df['weekday'].value_counts().reindex(weekday_order)

    plt.figure(figsize=(8, 5))
    weekday_counts.plot(kind='bar')
    plt.title('요일별 리뷰 개수')
    plt.xlabel('요일')
    plt.ylabel('리뷰 개수')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()