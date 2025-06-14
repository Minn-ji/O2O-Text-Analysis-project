import matplotlib.pyplot as plt

# 월 별 리뷰 수 그래프 작성

def plot_monthly_review_counts(monthly_counts):
    monthly_counts['year_month'] = monthly_counts['year_month'].astype(str)

    plt.figure(figsize=(16, 5))
    plt.plot(monthly_counts['year_month'], monthly_counts['review_count'], marker='o')
    plt.title('월별 리뷰 개수 추이')
    plt.xlabel('월(YYYY-MM)')
    plt.ylabel('리뷰 개수')
    plt.xticks(rotation=45, fontsize=9)
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()