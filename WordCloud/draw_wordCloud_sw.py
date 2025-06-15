import os
from konlpy.tag import Okt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

from matplotlib import rcParams
rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False


def generate_wordcloud_sw(df, app_name, stop_list=None):
    # 형태소 분석기
    okt = Okt()

    # 명사 제외하고 동사/형용사/부사 등만 추출
    tokens = []
    for text in df["text"]:
        pos_tags = okt.pos(text, stem=True)  # 원형 보존
        tokens += [word for word, tag in pos_tags if tag not in ['Noun', 'Josa', 'Punctuation', 'Foreign', 'Alpha']]

    # 불용어 제거
    if stop_list==None:
        stopwords = ['하다', '되다', '있다', '없다', '되었', '해주다']
    else:
        stopwords = stop_list
    tokens = [word for word in tokens if word not in stopwords and len(word) > 1]

    # 단어 빈도수 계산
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
    plt.title("명사 제외 워드클라우드", fontsize=18)
    plt.savefig(f'result/{app_name}_wordcloud.png')
    print(f'{app_name} word cloud 생성 완료!')