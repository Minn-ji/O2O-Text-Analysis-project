import pandas as pd

# 특정 일자 리뷰 추출

def get_reviews_by_day(df, year, month, day):
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    mask = (
        (df['date'].dt.year == year) &
        (df['date'].dt.month == month) &
        (df['date'].dt.day == day)
    )
    return df.loc[mask].reset_index(drop=True)