import matplotlib.pyplot as plt
import pandas as pd

# 탐색된 이상치 그래프 작성

def plot_daily_reviews_with_outliers(df, years=None, method='iqr', thresh=1.5):
    import matplotlib.pyplot as plt

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

        vals = daily_counts.values

        if method == 'iqr':
            q1 = pd.Series(vals).quantile(0.25)
            q3 = pd.Series(vals).quantile(0.75)
            iqr = q3 - q1
            lower = q1 - thresh * iqr
            upper = q3 + thresh * iqr
            outlier_idx = (vals < lower) | (vals > upper)
        else:
            mean = vals.mean()
            std = vals.std()
            lower = mean - thresh * std
            upper = mean + thresh * std
            outlier_idx = (vals < lower) | (vals > upper)

        plt.plot(all_mmdd, vals, label=f'{year}')
        plt.scatter(pd.Series(all_mmdd)[outlier_idx], vals[outlier_idx], color='red', s=40, zorder=10, label=f'{year} 이상치')

    plt.title('연도별 일별 리뷰 개수 (이상치 표시)')
    plt.xlabel('날짜 (MM-DD)')
    plt.ylabel('리뷰 개수')
    plt.legend()
    plt.xticks(rotation=45, fontsize=8, ticks=range(0, 366, 15))
    plt.tight_layout()
    plt.grid(alpha=0.2)
    plt.show()