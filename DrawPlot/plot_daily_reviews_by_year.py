import matplotlib.pyplot as plt
import pandas as pd

# 특정 연도의 일 별 리뷰 수 그래프 작성

def plot_daily_reviews_by_year(df, years=None):

    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    if years is None:
        years = sorted(df['date'].dt.year.unique())
    
    plt.figure(figsize=(18, 5))

    for year in years:
        year_df = df[df['date'].dt.year == year]
        daily_counts = year_df['date'].dt.strftime('%m-%d').value_counts().sort_index()
        all_days = pd.date_range(f'{year}-01-01', f'{year}-12-31')
        all_mmdd = all_days.strftime('%m-%d')
        daily_counts = daily_counts.reindex(all_mmdd, fill_value=0)
        plt.plot(all_mmdd, daily_counts.values, label=str(year))
    
    plt.title('연도별 일별 리뷰 개수')
    plt.xlabel('날짜 (MM-DD)')
    plt.ylabel('리뷰 개수')
    plt.legend()
    plt.xticks(rotation=45, fontsize=8, ticks=range(0, 366, 15), labels=all_mmdd[::15])
    plt.tight_layout()
    plt.grid(alpha=0.2)
    plt.show()