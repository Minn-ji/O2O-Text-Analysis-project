from google_play_scraper import Sort, reviews
import pandas as pd
from datetime import datetime, timedelta
import re

#  앱 패키지 이름
app_package = 'net.skyscanner.android.main'

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
        lang='ko',        # 리뷰 UI가 한국어 기준
        country='kr',     # 한국 Play Store
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

#  DataFrame 생성
df = pd.DataFrame(all_reviews)

#  전처리: 결측 제거, 중복 제거
df = df.dropna(subset=["text"])
df["text"] = df["text"].apply(lambda x: x.strip())
df = df[df["text"] != ""]
df = df.drop_duplicates(subset=["text"])

# 텍스트 정제 함수 (특수문자 제거, 공백 정리, 영어 제거)
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)                          # 연속 공백을 하나로
    text = re.sub(r'[a-zA-Z]', '', text)                      # 영어 제거
    text = re.sub(r'[^\w\sㄱ-ㅎ가-힣]', '', text)             # 한글, 숫자, 언더바만 남김
    return text.strip()

# 텍스트 정제
df["text"] = df["text"].apply(clean_text)

# 한글 포함된 리뷰만 필터링
df = df[df["text"].apply(lambda x: bool(re.search(r'[가-힣]', x)))]

# 날짜 기준 정렬 및 컬럼 순서 지정
df = df.sort_values("date")
df = df[["date", "score", "text"]]


# 오타 사전 정의
typo_dict = {
    # 굳 계열
    r'(귲|굿|구구구구수두구수구굿|구굳|굿굿국굿굿굿굿굿굿|궁|구수구구구구굿|궇|군|귣|그구긋|구우웃|귯|긋귿굿굳|굿좝|굿잡|구구굿|굳뜨|구웃|둣|구구우웃|개굳|개굿|굿구슥ㆍ|구굿구굿|긋잡|굿뜨|구뜨)': '굳',
    
    # 좋아 계열
    r'(조아|좋으|져아|좋슴다|좋앙|좋노|조하|종사|좋ㄷㅏ|쩡좋음|좋앙ᆢ|쪼아|조음|좋와)': '좋아',
    
    # 괜찮다 계열
    r'(괘추|괘안음|갠춘)': '괜찮다',
    
    # 최고 계열
    r'(초고|쵝오|최공)': '최고',
    
    # 욕설
    r'(씨\s*ㅡ\s*발)': '욕설',
    
    # 좋아요 계열
    r'(좋ㅇㅏ요|좋마요|좋시요|좋ㅇㄱㆍ요|좋아요용|좋ㅇ요|종\s*ㅣ\s*ᆞ요|좋인요|좋아욕|좋아욥|조와용|조으네요|조으다|조으당|조아연|조아여|존좋텡\s*ㅜㅜ)': '좋아요',
    
    # 좋은 계열
    r'(존)': '좋은',

    # 추천 계열
    r'(추찬)': '추천',

    # 꿀 계열
    r'(갸꿀)': '꿀',

    # 최저가 계열
    r'(최조가)': '최저가'
}

# 오타 정제 함수
def normalize_typos(text):
    for pattern, replacement in typo_dict.items():
        text = re.sub(pattern, replacement, text)
    return text