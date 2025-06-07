import os
import torch
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from util.basic_tools import load_tokenizer, load_model, load_device, get_sentiment_map

def predict_sentiment(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs, dim=-1).item()

    return label

def replace_sentiment_label_to_score(df):
    # 각 감정 레이블을 유저 감정에 대응하는 별점 스케일(1~5)로 매핑하여 비교
    # 그 후 감정 기반 점수(emotion_score)와 실제 별점(rating)을 수치적으로 비교 → 정량적 성능 평가 가능
    # 감정 ID → 스케일된 점수
    emotion_to_score = {
        0: 5,
        1: 4.5,
        2: 4,
        3: 5,
        4: 5,
        5: 3,
        6: 3,
        7: 2,
        8: 1.5,
        9: 1,
        10: 2
    }

    df["emotion_score"] = df["sentiment"].map(emotion_to_score)
    # 평균 감정 점수
    mean_emotion_score = df['emotion_score'].mean()
    # 감성 점수 (0~100 정규화)
    sentiment_score = ((mean_emotion_score - 1) / 4) * 100
    print(f"앱 감성 점수: {sentiment_score:.1f}/100")

    # 평가
    mae = (df["emotion_score"] - df["score"]).abs().mean()
    mse = mean_squared_error(df["score"], df["emotion_score"])
    corr, _ = pearsonr(df["score"], df["emotion_score"])

    print(f"MAE: {mae:.3f}, MSE: {mse:.3f}, Pearson Correlation: {corr:.3f}")


def make_json_file(df, tokenizer, model, device, sentiment_map):
    result_dict = []
    for sentence in df['reviews']:
        pred = predict_sentiment(sentence, tokenizer, model, device)
        result = {'review': sentence, 'sentiment': sentiment_map[pred], 'sentiment_number': pred}
        result_dict.append(result)

    return result_dict


def make_sentiment_columns(df, tokenizer, model, device, app_name='kakao_taxi'):
    df['sentiment'] = df['reviews'].apply(lambda x: predict_sentiment(x, tokenizer, model, device))

    os.makedirs('result', exist_ok=True)
    df.to_csv(f'result/{app_name}_sentiment_analyzed.csv', index=False)
    print(f'result/{app_name}_sentiment_analyzed.csv 저장완료!')



if __name__ == '__main__': # 상단에서 모듈로 돌려야함
    tokenizer = load_tokenizer()
    model = load_model()
    device = load_device()
    senti_label = predict_sentiment('오늘 너무 피곤하다.', tokenizer, model, device)
    sentiment_map = get_sentiment_map()
    print(sentiment_map[senti_label])

