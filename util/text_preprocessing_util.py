import os
import re
import pandas as pd
import numpy as np
from .basic_tools import load_kiwi, load_typo_dict

def remove_tag(text):
    if isinstance(text, str):
        p = re.compile(r'<[^>]+>')
        cleaned = p.sub('', text)
        cleaned = cleaned.replace('\r', '').replace('\n', '').replace('\t', '')
        return cleaned.strip()
    else:
        return str(text).strip()

def ko_preprocessing(text): # 상원님 코드 병합 완료
    text = re.sub(r'\s+', ' ', text)    # 연속 공백을 하나로
    clean_text = re.sub(r"[^가-힣\s]", "", text) # 초성 제외 한글만
    return clean_text.strip()

def normalize_typos(text, typo_dict): # 직접 찾은 오탈자들 표준어로 대체
    for correct, typos in typo_dict.items():
        for typo in typos:
            pattern = re.compile(re.escape(typo))
            text = pattern.sub(correct, text)

    return text

def preprocess_DataFrame(df, kiwi, typo_dict, app_name):
    df['date'] = pd.to_datetime(df['at'].apply(lambda x: x[:10]), format='%Y-%m-%d')
    df['text'] = df['text'].apply(lambda x: ko_preprocessing(remove_tag(x)))
    df['text'] = df['text'].apply(lambda x: np.nan if x.strip() == '' else x.strip())
    df = df.dropna().reset_index(drop=True)
    df['text'] = df['text'].apply(lambda x: kiwi.space(x))
    df['text'] = df['text'].apply(lambda x: normalize_typos(x, typo_dict))

    df = df[["date", "score", "text"]]
    df = df.sort_values("date")

    os.makedirs('result', exist_ok=True)
    df.to_csv(f'result/{app_name}_google_store_scraper_10years_preprocessed.csv', index=False, escapechar='\\')
    print(f"완료! 전처리된 {app_name} 리뷰가 CSV로 저장되었습니다.")


if __name__ == '__main__': # 상단에서 모듈로 돌려야함
    df = pd.read_csv('result/yogiyo_google_store_scraper_10years.csv')
    typo_dict = load_typo_dict()
    kiwi = load_kiwi()
    preprocess_DataFrame(df, kiwi, typo_dict, 'yogiyo')
