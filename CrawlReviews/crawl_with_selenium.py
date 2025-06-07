import os
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import time
import pandas as pd
import requests
import xml.etree.ElementTree as ET


def crawl_google_store_review_with_selenium(url, app_name):
    driver = webdriver.Chrome()
    driver.implicitly_wait(3)

    # scraping하려는 웹페이지 주소를 get()에 전달
    driver.get(url)

    time.sleep(2)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 3);")
    review_button = driver.find_element(By.XPATH,
                                        '/html/body/c-wiz[2]/div/div/div[1]/div/div[2]/div/div[1]/div[1]/c-wiz[5]/section/header/div/div[2]/button')
    review_button.click()

    review_results = []
    star_results = []
    date_results = []
    for i in tqdm(range(10000), total=10000, desc='crawling'):
        try:
            review_text = driver.find_element(By.XPATH,
                                              f'/html/body/div[5]/div[2]/div/div/div/div/div[2]/div/div[2]/div[{i + 1}]/div[1]')
            star_text = driver.find_element(By.XPATH,
                                            f'/html/body/div[5]/div[2]/div/div/div/div/div[2]/div/div[2]/div[{i + 1}]/header/div[2]/div')
            star_text = star_text.get_attribute("aria-label")
            date_text = driver.find_element(By.XPATH,
                                            f'/html/body/div[5]/div[2]/div/div/div/div/div[2]/div/div[2]/div[{i + 1}]/header/div[2]/span')

            review_results.append(review_text.text)
            star_results.append(star_text)
            date_results.append(date_text.text)
            # print(review_text.text)
            # print(star_text)
            # print(date_text.text)
        except NoSuchElementException:
            scrollable_div = driver.find_element(By.CSS_SELECTOR, 'div.fysCi')
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
            time.sleep(6)
            continue
        except TimeoutException:
            time.sleep(60)
            continue

    review_df = pd.DataFrame({"date": date_results, "score": star_results, 'text': review_results})
    save_result(app_name, review_df)

    return review_df


def crawl_app_store_review(url, app_name):
    # RSS 피드 가져오기
    response = requests.get(url)
    response.raise_for_status()  # 요청 실패 시 예외 발생

    # XML 파싱
    root = ET.fromstring(response.content)

    # 네임스페이스 정의
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'im': 'http://itunes.apple.com/rss'
    }

    # 리뷰 데이터 추출
    reviews = []
    for entry in root.findall('atom:entry', ns):
        # 첫 번째 entry는 앱 정보이므로 건너뜀
        if entry.find('atom:id', ns).text.endswith('/id368677368'):
            continue

        title = entry.find('atom:title', ns).text
        content = entry.find('atom:content', ns).text
        updated = entry.find('atom:updated', ns).text
        rating = entry.find('im:rating', ns).text

        reviews.append({
            'title': title,
            'content': content,
            'date': updated,
            'score': rating
        })

    review_df = pd.DataFrame(reviews)
    save_app_store_result(app_name, review_df)

def save_result(app_name, review_df):
    prefix = app_name.replace(' ', '_')
    os.makedirs('result', exist_ok=True)
    review_df.to_csv(f'result/{prefix}_google_store.csv', index=False, escapechar='\\')
    print(f'save {prefix} result.')

def save_app_store_result(app_name, review_df):
    prefix = app_name.replace(' ', '_')
    os.makedirs('result', exist_ok=True)
    review_df.to_csv(f'result/{prefix}_app_store.csv', index=False, escapechar='\\')
    print(f'save {prefix} result.')


if __name__ == '__main__':
    # google store
    kakao_taxi_google_store = 'https://play.google.com/store/apps/details?id=com.kakao.taxi&hl=ko'
    kakao_df = crawl_google_store_review_with_selenium(kakao_taxi_google_store, 'kakao_taxi')

    uber_taxi_google_store = 'https://play.google.com/store/apps/details?id=com.ubercab&hl=ko'
    crawl_google_store_review_with_selenium(uber_taxi_google_store, 'uber_taxi')

    # app store
    # kakao_taxi_app_store = "https://itunes.apple.com/kr/rss/customerreviews/id=981110422/xml"
    # crawl_app_store_review(kakao_taxi_app_store, 'kakao_taxi')

    # uber_taxi_app_store = "https://itunes.apple.com/kr/rss/customerreviews/id=368677368/xml"
    # crawl_app_store_review(uber_taxi_app_store, 'uber_taxi')
