import matplotlib.pyplot as plt

# 점수 분포 시각화

def plot_score_distribution(df, score_column='score'):
    score_counts = df[score_column].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(score_counts.index.astype(str), score_counts.values, color='lightgreen')
    plt.title('리뷰 점수 분포')
    plt.xlabel('점수')
    plt.ylabel('리뷰 개수')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()