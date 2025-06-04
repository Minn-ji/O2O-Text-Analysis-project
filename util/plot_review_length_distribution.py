import matplotlib.pyplot as plt

# 리뷰 길이 분포 분석

def plot_review_length_distribution(df, text_column='content'):
    df['length'] = df[text_column].astype(str).str.len()

    plt.figure(figsize=(10, 5))
    plt.hist(df['length'], bins=50, color='skyblue', edgecolor='black')
    plt.title('리뷰 길이 분포')
    plt.xlabel('글자 수')
    plt.ylabel('리뷰 개수')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()