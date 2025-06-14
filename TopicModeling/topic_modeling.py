import pandas as pd
from gensim import corpora
from kiwipiepy import Kiwi

def merge_text(df, app_name):
    df = df[df['sentiment'].isin([7, 8, 9, 10])] # 부정 키워드만 추출
    merged_text = ''
    for text in df['text']:
        if pd.notnull(text):
            merged_text += str(text) + '. '
    with open(f'result/{app_name}_review_merged.txt', 'w', encoding='utf-8') as f:
        f.write(merged_text)
    print('저장 완료!')

def load_docs(app_name):
    with open(f'result/{app_name}_review_merged.txt', 'r', encoding='utf-8') as f:
        docs = f.read()
    return docs

def return_merge_df():
    kakao_taxi_docs = load_docs('kakao_taxi')
    uber_taxi_docs = load_docs('uber_taxi')
    yogiyo_docs = load_docs('yogiyo')
    yeogi_docs = load_docs('yeogi')
    skyscanner_docs = load_docs('skyscanner')
    nol_docs = load_docs('nol')
    df = {
        'kakao_taxi': kakao_taxi_docs,
        'uber_taxi': uber_taxi_docs,
        'yogiyo': yogiyo_docs,
        'yeogi': yeogi_docs,
        'skyscanner': skyscanner_docs,
        'nol': nol_docs
    }

    df = pd.DataFrame([
        {'app_name': k, 'review': v}
        for k, v in df.items()
    ])
    df.to_csv('result/merged_review_for_topic_modeling.csv', index=False)

def generate_word_dict_corpus():
    df = pd.read_csv('result/merged_review_for_topic_modeling.csv')
    kiwi = Kiwi()
    ALLOW_TAGS = {'NNG', 'SL', 'XR'}

    df['review_preprocessed'] = df['review'].apply(lambda x: kiwi.tokenize(x))
    df['review_preprocessed'] = df['review_preprocessed'].apply(
        lambda tokens: [token.form for token in tokens if token.tag in ALLOW_TAGS]
    )

    # 2. 전처리: 너무 짧은 문장 제거
    df = df[df['review_preprocessed'].apply(lambda x: len(x) >= 2)].copy()

    # 3. Dictionary & Corpus 생성
    word_dict = corpora.Dictionary(df['review_preprocessed'])
    word_dict.filter_extremes(no_below=2, no_above=0.95)  # 단어 수 기준 필터
    word_dict.save('result/word_dict.dict')
    # word_dict = corpora.Dictionary.load('result/word_dict.dict')
    corpus = [word_dict.doc2bow(review) for review in df['review_preprocessed']]
    corpora.MmCorpus.serialize('result/corpus.mm', corpus)
    # corpus = corpora.MmCorpus('result/corpus.mm')
    print('생성 및 저장 완료! \n', [(idx,word) for idx, word in word_dict.items()])
