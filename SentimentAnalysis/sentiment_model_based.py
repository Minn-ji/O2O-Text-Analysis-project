import os
import torch
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, classification_report
from util.basic_tools import load_tokenizer, load_model, load_device, get_sentiment_map

def predict_sentiment(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs, dim=-1).item()

    return label
def map_score_to_polarity(score):
    if score <= 2:
        return "negative"
    elif score == 3:
        return "neutral"
    else:
        return "positive"

def map_sentiment_id_to_polarity(sentiment):
    if sentiment in [0, 1, 2, 3, 4]:
        return "positive"
    elif sentiment in [5, 6]:
        return "neutral"
    else:
        return "negative"

def replace_sentiment_label_to_score(df):
    true_polarity = df["score"].apply(map_score_to_polarity)
    predicted_polarity = df["sentiment"].apply(map_sentiment_id_to_polarity)

    # 정확도 및 F1 등 평가
    print("Accuracy:", accuracy_score(true_polarity, predicted_polarity))
    print(classification_report(true_polarity, predicted_polarity, digits=3))

def make_json_file(df, tokenizer, model, device, sentiment_map):
    result_dict = []
    for sentence in df['text']:
        pred = predict_sentiment(sentence, tokenizer, model, device)
        result = {'text': sentence, 'sentiment': sentiment_map[pred], 'sentiment_number': pred}
        result_dict.append(result)

    return result_dict


def make_sentiment_columns(df, tokenizer, model, device, app_name='kakao_taxi'):
    df['sentiment'] = df['text'].apply(lambda x: predict_sentiment(x, tokenizer, model, device))

    os.makedirs('result', exist_ok=True)
    df.to_csv(f'result/{app_name}_sentiment_analyzed_with_model.csv', index=False)
    print(f'result/{app_name}_sentiment_analyzed_with_model.csv 저장완료!')



if __name__ == '__main__': # 상단에서 모듈로 돌려야함
    tokenizer = load_tokenizer()
    model = load_model()
    device = load_device()
    senti_label = predict_sentiment('오늘 너무 피곤하다.', tokenizer, model, device)
    sentiment_map = get_sentiment_map()
    print(sentiment_map[senti_label])

