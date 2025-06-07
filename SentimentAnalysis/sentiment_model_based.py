import os
import torch
from util.basic_tools import load_tokenizer, load_model, load_device, get_sentiment_map

def predict_sentiment(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs, dim=-1).item()

    return label

def replace_sentiment_label_to_rating():
    print('')


def make_json_file(df, tokenizer, model, device, sentiment_map):
    result_dict = []
    for sentence in df['reviews']:
        pred = predict_sentiment(sentence, tokenizer, model, device)
        result = {'review': sentence, 'sentiment': sentiment_map[pred], 'sentiment_number': pred}
        result_dict.append(result)

    os.makedirs('result', exist_ok=True)
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

