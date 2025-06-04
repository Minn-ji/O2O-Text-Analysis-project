import pandas as pd

# 이상치 탐색

def detect_daily_review_outliers(df, years=None, method='iqr', thresh=1.5):
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    if years is None:
        years = sorted(df['date'].dt.year.unique())

    outlier_dates = []

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

        outlier_days = [f'{year}-{d}' for i, d in enumerate(all_mmdd) if outlier_idx[i]]
        outlier_counts = vals[outlier_idx]

        for date, cnt in zip(outlier_days, outlier_counts):
            outlier_dates.append((date, cnt))

    outlier_df = pd.DataFrame(outlier_dates, columns=['date', 'review_count'])
    return outlier_df