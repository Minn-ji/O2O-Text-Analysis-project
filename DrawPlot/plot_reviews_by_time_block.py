import matplotlib.pyplot as plt
import pandas as pd

# 시간대 별 리뷰 작성

def plot_reviews_by_time_block(df):
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    def classify_time_block(hour):
        if 5 <= hour < 11:
            return '아침'
        elif 11 <= hour < 17:
            return '낮'
        elif 17 <= hour < 21:
            return '저녁'
        else:
            return '밤'

    df['hour'] = df['date'].dt.hour
    df['time_block'] = df['hour'].apply(classify_time_block)

    block_order = ['아침', '낮', '저녁', '밤']
    block_counts = df['time_block'].value_counts().reindex(block_order)

    plt.figure(figsize=(8, 5))
    plt.bar(block_counts.index, block_counts.values)
    plt.title('시간대별 리뷰 개수 (아침/낮/저녁/밤)')
    plt.xlabel('시간대')
    plt.ylabel('리뷰 개수')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()