import os
import pandas as pd
from datetime import datetime, timedelta
from google_play_scraper import Sort, reviews

def crawl_google_store_review_with_scraper(app_package, app_name):
    #  수집 대상: 10년 전까지
    end_date = datetime.today() - timedelta(days=365 * 10)

    #  수집용 리스트 및 토큰 초기화
    all_reviews = []
    next_token = None
    page = 0

    while True:
        #  100개씩 최신순으로 리뷰 요청
        result, next_token = reviews(
            app_package,
            lang='ko',  # 리뷰 UI가 한국어 기준
            country='kr',  # 한국 Play Store
            sort=Sort.NEWEST,
            count=100,
            continuation_token=next_token
        )

        for r in result:
            review_date = r['at']
            if review_date < end_date:
                break  # 10년 이전이면 중단

            all_reviews.append({
                'date': review_date,
                'score': r['score'],
                'text': r['content'].strip()
            })

        page += 1
        print(f" {len(all_reviews)} reviews collected... (Page {page})")

        # 다음 페이지 없거나 10년 초과되면 종료
        if not next_token or review_date < end_date:
            break

    df = pd.DataFrame(all_reviews)

    os.makedirs('result', exist_ok=True)
    df.to_csv(f'result/{app_name}_google_store_scraper_10years.csv', index=False, escapechar='\\')
    print(f"완료! {app_name} 리뷰가 CSV로 저장되었습니다.")


if __name__ == '__main__':# 상단에서 모듈로 돌려야함
    app_package, app_name = 'net.skyscanner.android.main', 'skyscanner'
    crawl_google_store_review_with_scraper(app_package, app_name)

    # app_package, app_name = 'com.ubercab', 'uber_taxi'
    # app_package, app_name = 'com.kakao.taxi', 'kakao_taxi'