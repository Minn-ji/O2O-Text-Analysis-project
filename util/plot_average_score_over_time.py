import matplotlib.pyplot as plt
import pandas as pd

# 평균 점수 추이 (연도/월 기준)

def plot_average_score_over_time(df, score_column='score', by='month'):
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    if by == 'month':
        df['period'] = df['date'].dt.to_period('M')
    elif by == 'year':
        df['period'] = df['date'].dt.year
    else:
        raise ValueError("`by` must be 'month' or 'year'")

    avg_score = df.groupby('period')[score_column].mean()

    plt.figure(figsize=(12, 5))
    plt.plot(avg_score.index.astype(str), avg_score.values, marker='o', color='purple')
    plt.title(f'{by}별 평균 리뷰 점수')
    plt.xlabel(by)
    plt.ylabel('평균 점수')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
