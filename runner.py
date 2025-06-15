import pandas as pd

from util.basic_tools import *
from util.text_preprocessing_util import preprocess_DataFrame
from CrawlReviews.crawl_with_selenium import crawl_google_store_review_with_selenium
from CrawlReviews.crawl_with_scraper import crawl_google_store_review_with_scraper
from SentimentAnalysis.sentiment_model_based import make_json_file, make_sentiment_columns, replace_sentiment_label_to_score
from WordCloud.draw_wordCloud import generate_wordcloud
from WordCloud.draw_wordCloud_sw import generate_wordcloud_sw
from TopicModeling.topic_modeling import return_merge_df, generate_word_dict_corpus, get_topic_ratio_for_each_document, find_best_topic_count

def scraper_crawl_google_store_review(app_package, app_name):
    # 구글 스토어 스크래퍼
    crawl_google_store_review_with_scraper(app_package, app_name)

def selenium_crawl_google_store_review(url, app_name):
    # 셀레니움 구글 스토어
    crawl_google_store_review_with_selenium(url, app_name)

def preprocess_review_df(df, app_name):
    typo_dict = load_typo_dict()
    kiwi = load_kiwi()
    preprocess_DataFrame(df, kiwi, typo_dict, app_name)

def get_sentiment_column(app_name):
    tokenizer = load_tokenizer()
    model = load_model()
    device = load_device()
    make_sentiment_columns(pre_processed_df, tokenizer, model, device, app_name)
    # 모델 성능 측정
    senti_added_df = pd.read_csv(f'result/{app_name}_sentiment_analyzed_with_model.csv')
    replace_sentiment_label_to_score(senti_added_df)

def get_sentiment_analysis_result_with_model(pre_processed_df, app_name):
    # json으로 결과 보기 위한 용도. 안 돌려도 됨
    tokenizer = load_tokenizer()
    model = load_model()
    device = load_device()
    sentiment_map = get_sentiment_map()
    result_dict = make_json_file(pre_processed_df, tokenizer, model, device, sentiment_map)
    save_json(f"result/{app_name}_sentiment_result.json", result_dict)

def get_wordcloud(preprocessed_df, app_name):
    generate_wordcloud(preprocessed_df, app_name)
    # generate_wordcloud_sw(preprocessed_df, app_name=app_name)

def merge_reviews_for_topic_modeling(kakao_df, uber_df, yogiyo_df, yeogi_df, skyscanner_df, nol_df):
    return_merge_df(kakao_df, uber_df, yogiyo_df, yeogi_df, skyscanner_df, nol_df)

def topic_modeling_():
    generate_word_dict_corpus()
    get_topic_ratio_for_each_document()
    find_best_topic_count()

if __name__ == '__main__': # python -m runner로 실행 (모듈로)
    
    ###---- [예시 데이터 : kakao_taxi] ----###
    ## 1. 크롤링
    app_package, app_name = 'com.kakao.taxi', 'kakao_taxi'
    scraper_crawl_google_store_review(app_package, app_name)

    ## 2. 전처리
    kakao_raw_df = pd.read_csv(f'result/{app_name}_google_store_scraper_10years.csv')
    preprocess_review_df(kakao_raw_df, app_name=app_name)

    ## 3-1. 모델 기반 감성 분석 (사전 기반은 도메인 내부에서 실행)
    get_sentiment_column(app_name=app_name)
    
    ## 3-2. 워드클라우드 생성 (이상치에 대한 wc는 도메인 내부 ipynb로 실행)
    preprocessed_kakao_df = pd.read_csv(f'result/{app_name}_google_store_scraper_10years_preprocessed.csv')
    get_wordcloud(preprocessed_kakao_df, app_name=app_name)

    ## 3-3. 토픽 모델링 word_dict, corpus, trained lda model 생성 (kakao 포함 전부 감정분석이 완료되어야 수행 가능)(결과는 ipynb에서 확인)
    kakao_df = pd.read_csv('result/kakao_taxi_sentiment_analyzed_with_model.csv')
    uber_df = pd.read_csv('result/uber_taxi_sentiment_analyzed_with_model.csv')
    yogiyo_df = pd.read_csv('result/yogiyo_sentiment_analyzed_with_model.csv')
    skyscanner_df = pd.read_csv('result/skyscanner_sentiment_analyzed_with_model.csv')
    yeogi_df = pd.read_csv('result/yeogi_sentiment_analyzed_with_model.csv')
    nol_df = pd.read_csv('result/nol_sentiment_analyzed_with_model.csv')
    merge_reviews_for_topic_modeling(kakao_df, uber_df, yogiyo_df, yeogi_df, skyscanner_df, nol_df)
    topic_modeling_()

