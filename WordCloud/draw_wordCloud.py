import os

from kiwipiepy import Kiwi
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rcParams
rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False
def generate_wordcloud(df, app_name):
    # Kiwi 형태소 분석기 초기화
    kiwi = Kiwi()

    # 명사 제외하고 동사/형용사/부사 등만 추출
    tokens = []
    for text in df["text"].dropna():  # 결측치 방지
        analyzed = kiwi.analyze(text)
        for sentence in analyzed:
            for word, tag, _, _ in sentence[0]:  # 형태소 정보만 추출
                if tag in ['XR']: #  not in ['NNG', 'NNP', 'NNB', 'NP', 'NR', 'JKS', 'JKC', 'JKG', 'JKB', 'JKV', 'JKQ', 'JC', 'SF', 'SE', 'SSO', 'SSC', 'SC', 'SY', 'SH', 'SL']:
                    # 안될경우 tag in ['XR']:
                    tokens.append(word)


    # 불용어 제거
    stopwords = ['하다', '되다', '있다', '없다', '해주다', '같다', '그렇다', '는데']
    tokens = [word for word in tokens if word not in stopwords and len(word) > 1]

    # 단어 빈도 계산
    word_freq = Counter(tokens)

    # 워드클라우드 생성
    font_path = "assets/malgun.ttf"
    wc = WordCloud(
        font_path=font_path,
        background_color='white',
        width=800,
        height=600
    ).generate_from_frequencies(word_freq)
    os.makedirs('result', exist_ok=True)
    # 시각화
    plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("명사 제외 워드클라우드 (Kiwi 기반)", fontsize=18)

    plt.savefig(f'result/{app_name}_wordcloud.png')
    print(f'{app_name} word cloud 생성 완료!')