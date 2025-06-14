import pandas as pd

from util.basic_tools import *
from util.text_preprocessing_util import preprocess_DataFrame
# from CrawlReviews.crawl_with_selenium import crawl_google_store_review_with_selenium
# from CrawlReviews.crawl_with_scraper import crawl_google_store_review_with_scraper
from SentimentAnalysis.sentiment_model_based import make_json_file, make_sentiment_columns, replace_sentiment_label_to_score
from WordCloud.draw_wordCloud import generate_wordcloud
from WordCloud.draw_wordCloud_sw import generate_wordcloud_sw

#def scraper_crawl_google_store_review(app_package, app_name):
    # 구글 스토어 스크래퍼
    # crawl_google_store_review_with_scraper(app_package, app_name)

# def selenium_crawl_google_store_review(url, app_name):
#     # 셀레니움 구글 스토어
#     crawl_google_store_review_with_selenium(url, app_name)

# 여기부터 실행
def preprocess_review_df(df, app_name):
    # 자동 저장됨
    typo_dict = load_typo_dict()
    kiwi = load_kiwi()
    preprocess_DataFrame(df, kiwi, typo_dict, app_name)

def get_sentiment_column(pre_processed_df, app_name):
    tokenizer = load_tokenizer()
    model = load_model()
    device = load_device()
    make_sentiment_columns(pre_processed_df, tokenizer, model, device, app_name)
    senti_added_df = pd.read_csv(f'result/{app_name}_sentiment_analyzed_with_model.csv')
    replace_sentiment_label_to_score(senti_added_df) # 성능 보여줌

# def get_sentiment_analysis_result_with_model(pre_processed_df, app_name):
#     # json으로 결과 보기 위한 용도. 안 돌려도 됨
#     tokenizer = load_tokenizer()
#     model = load_model()
#     device = load_device()
#     sentiment_map = get_sentiment_map()
#     result_dict = make_json_file(pre_processed_df, tokenizer, model, device, sentiment_map)
#     save_json(f"result/{app_name}_sentiment_result.json", result_dict)

def get_wordcloud(preprocessed_df, app_name):
    # generate_wordcloud(preprocessed_df, app_name)
    generate_wordcloud_sw(preprocessed_df, app_name)

if __name__ == '__main__': # python -m runner로 실행 (모듈로)
    # preprocessed_kakao = pd.read_csv('assets/kakao_taxi_store_merged_scraper_10years_preprocessed.csv')
    # get_sentiment_column(preprocessed_kakao, 'kakao_taxi')
    # get_wordcloud(preprocessed_kakao, 'kakao_taxi')
    #
    # preprocessed_uber = pd.read_csv('assets/uber_taxi_store_merged_scraper_10years_preprocessed.csv')
    # get_wordcloud(preprocessed_uber, 'uber_taxi')

    preprocessed_skyscanner = pd.read_csv('assets/skyscanner_reviews_korean_last.csv')
    get_wordcloud(preprocessed_skyscanner, 'skyscanner')

