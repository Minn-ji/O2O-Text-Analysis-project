import pandas as pd

# 월 별 리뷰 수 계산

def count_reviews_per_month(df):

    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    df['year_month'] = df['date'].dt.to_period('M')

    monthly_counts = df.groupby('year_month').size().reset_index(name='review_count')
    return monthly_counts