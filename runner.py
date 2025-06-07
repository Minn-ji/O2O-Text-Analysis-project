from util.basic_tools import *
from util.text_preprocessing_util import preprocess_DataFrame
from CrawlReviews.crawl_with_selenium import crawl_google_store_review_with_selenium
from CrawlReviews.crawl_with_scraper import crawl_google_store_review_with_scraper
from SentimentAnalysis.sentiment_model_based import make_json_file, make_sentiment_columns

def scraper_crawl_google_store_review(app_package, app_name):
    # 구글 스토어 스크래퍼
    crawl_google_store_review_with_scraper(app_package, app_name)

def selenium_crawl_google_store_review(url, app_name):
    # 셀레니움 구글 스토어
    crawl_google_store_review_with_selenium(url, app_name)

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

def get_sentiment_analysis_result_with_model(pre_processed_df, app_name):
    # json으로 결과 보기 위한 용도. 안 돌려도 됨
    tokenizer = load_tokenizer()
    model = load_model()
    device = load_device()
    sentiment_map = get_sentiment_map()
    result_dict = make_json_file(pre_processed_df, tokenizer, model, device, sentiment_map)
    save_json(f"result/{app_name}_sentiment_result.json", result_dict)


if __name__ == '__main__':
    # uber_taxi_google_url = 'https://play.google.com/store/apps/details?id=com.ubercab&hl=ko'
    # app_name = 'uber_taxi'
    #
    # app_package, app_name = 'net.skyscanner.android.main', 'skyscanner'
    print()