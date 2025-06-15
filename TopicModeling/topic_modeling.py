import pandas as pd
import ast
from gensim import corpora, models
from kiwipiepy import Kiwi
import matplotlib.pyplot as plt

import gensim
from gensim.models import CoherenceModel

def preprocess_reviews(df, app_name):
    kiwi = Kiwi()
    df = df[df['sentiment'].isin([7, 8, 9, 10])]

    ALLOW_TAGS = {'NNG', 'VA', 'VV', 'XR', 'MAG'}

    df['tokenized_text'] = df['text'].dropna().apply(lambda x: kiwi.tokenize(x))
    df['tokenized_text'] = df['tokenized_text'].apply(
        lambda tokens: [token.form for token in tokens if token.tag in ALLOW_TAGS]
    )
    # 너무 짧은 문장 제거
    df = df[df['tokenized_text'].apply(lambda x: len(x) >= 2)].copy()

    df['app_name'] = app_name
    return df[['app_name', 'tokenized_text']]

def return_merge_df(kakao_df, uber_df, yogiyo_df, yeogi_df, skyscanner_df, nol_df):
    dfs = [
        preprocess_reviews(kakao_df, 'kakao_taxi'),
        preprocess_reviews(uber_df, 'uber_taxi'),
        preprocess_reviews(yogiyo_df, 'yogiyo'),
        preprocess_reviews(yeogi_df, 'yeogi'),
        preprocess_reviews(skyscanner_df, 'skyscanner'),
        preprocess_reviews(nol_df, 'nol')
    ]

    # 전체 통합
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv('result/merged_review_for_topic_modeling.csv', index=False)


def generate_word_dict_corpus():
    print('시작')
    df = pd.read_csv('result/merged_review_for_topic_modeling.csv')
    df['tokenized_text'] = df['tokenized_text'].apply(ast.literal_eval)
    print('## 리스트로 반환: ', type(df['tokenized_text'][0]))
    
    # dict / corpus 생성
    word_dict = corpora.Dictionary(df['tokenized_text'])
    word_dict.filter_extremes(no_below=2, no_above=0.95)  # 단어 수 기준 필터
    word_dict.save('result/word_dict.dict')
    corpus = [word_dict.doc2bow(review) for review in df['tokenized_text']]
    corpora.MmCorpus.serialize('result/corpus.mm', corpus)

    # 불러올 때
    # word_dict = corpora.Dictionary.load('result/word_dict.dict')
    # corpus = corpora.MmCorpus('result/corpus.mm')
    print('## word dict & corpus 생성 및 저장 완료! \n', [(idx,word) for idx, word in word_dict.items()])


def get_topic_ratio_for_each_document():
    df = pd.read_csv('result/merged_review_for_topic_modeling.csv')
    df['tokenized_text'] = df['tokenized_text'].apply(ast.literal_eval)
    app_names = df['app_name'].tolist()

    word_dict = corpora.Dictionary.load('result/word_dict.dict')
    corpus = corpora.MmCorpus('result/corpus.mm')
    N_TOPICS = 10
    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus, num_topics=N_TOPICS, id2word=word_dict
    )
    ldamodel.save('result/lda_model')
    results = []
    dominant_topics = []
    for i, topic_list in enumerate(ldamodel[corpus]):
        sorted_topic_list = sorted(topic_list, key = lambda x: x[1], reverse=True)
        most_important_topic, most_important_ratio = sorted_topic_list[0]
        print(f"문서 {i}의 주요 토픽은 {most_important_topic}번이며, 비중은 {most_important_ratio * 100:.2f}%")
        print(f"{i}번째 문서의 topic 비중: {topic_list}")
        results.append([i, app_names[i], most_important_topic, most_important_ratio, sorted_topic_list])
        dominant_topics.append((app_names[i], most_important_topic))
    
    df_result = pd.DataFrame(results, columns=['문서 번호', 'app_name', '주요 토픽', '비중', '전체 토픽 분포'])
    df_result.to_csv('result/topic_modeling_describe.csv', index=False)

    print("\n토픽별 키워드 ============================")
    for i in range(N_TOPICS):
        terms = ldamodel.show_topic(i, topn=10)
        print(f"토픽 {i}: {[term for term, prob in terms]}")
    
    topic_df = pd.DataFrame(dominant_topics, columns=['app_name', 'topic'])
    topic_dist = topic_df.groupby(['app_name', 'topic']).size().unstack(fill_value=0)
    topic_ratio = topic_dist.div(topic_dist.sum(axis=1), axis=0)
    topic_ratio.to_csv('result/topic_ratio_by_app.csv')
    print('\n앱별 토픽 비율 저장 완료: result/topic_ratio_by_app.csv')
    print('문서별 토픽 정보 저장 완료!: result/topic_modeling_describe.csv')


def find_best_topic_count():
    df = pd.read_csv('result/merged_review_for_topic_modeling.csv')
    df['tokenized_text'] = df['tokenized_text'].apply(ast.literal_eval)

    word_dict = corpora.Dictionary.load('result/word_dict.dict')
    corpus = corpora.MmCorpus('result/corpus.mm')
    topics_range = range(10, 55, 2)
    scores = []

    for k in topics_range:
        ldamodel = models.LdaModel(corpus, num_topics=k, id2word=word_dict)
        cm = CoherenceModel(model=ldamodel, texts=df['tokenized_text'], dictionary=word_dict, coherence='c_v')
        scores.append(cm.get_coherence())

    plt.plot(topics_range, scores)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score (C_v)")
    plt.title("Optimal Number of Topics based on C_v")
    plt.savefig('result/number_of_topic_graph.png')
    print("Coherence 그래프 저장 완료")
