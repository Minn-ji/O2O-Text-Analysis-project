import matplotlib.pyplot as plt

# 점수별 리뷰 길이 비교

def plot_review_length_by_score(df, text_column='text', score_column='score'):
    df['length'] = df[text_column].astype(str).str.len()
    avg_length = df.groupby(score_column)['length'].mean()

    plt.figure(figsize=(8, 5))
    avg_length.plot(kind='bar', color='salmon')
    plt.title('점수별 평균 리뷰 길이')
    plt.xlabel('리뷰 점수')
    plt.ylabel('평균 글자 수')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()