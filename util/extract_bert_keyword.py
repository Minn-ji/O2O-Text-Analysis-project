import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

def recover_korean(text):
    return text.encode('latin1').decode('utf-8')

def get_sentiment_map():
    id2label= {
        0: "\uae30\uc068(\ud589\ubcf5\ud55c)",
        1: "\uace0\ub9c8\uc6b4",
        2: "\uc124\ub808\ub294(\uae30\ub300\ud558\ub294)",
        3: "\uc0ac\ub791\ud558\ub294",
        4: "\uc990\uac70\uc6b4(\uc2e0\ub098\ub294)",
        5: "\uc77c\uc0c1\uc801\uc778",
        6: "\uc0dd\uac01\uc774 \ub9ce\uc740",
        7: "\uc2ac\ud514(\uc6b0\uc6b8\ud55c)",
        8: "\ud798\ub4e6(\uc9c0\uce68)",
        9: "\uc9dc\uc99d\ub0a8",
        10: "\uac71\uc815\uc2a4\ub7ec\uc6b4(\ubd88\uc548\ud55c)"
    }

    sentiment_map = {}
    for i, v in id2label.items():
        decoded = v.encode('utf-8').decode('unicode_escape')
        sentiment_map[i] = recover_korean(decoded)
    return sentiment_map


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs, dim=-1).item()
    
    return label

def make_json_file(df, name):
    result_dict = []
    for sentence in df['reviews']:
        pred = predict_sentiment(sentence)
        result = {'review': sentence, 'sentiment': sentiment_map[pred], 'sentiment_number': pred}
        result_dict.append(result)

    with open(f"result/{name}_sentiment_result.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print(f"result/{name}_sentiment_result.json file 생성 완료")


def make_sentiment_columns(df, app_name='kakao_taxi'):
    df['sentiment'] = df['reviews'].apply(predict_sentiment)
    os.makedirs('result', exist_ok=True)
    df.to_csv(f'result/{app_name}_sentiment_analyzed.csv', index=False)
    print(f'result/{app_name}_sentiment_analyzed.csv 저장완료!')
    
    return df