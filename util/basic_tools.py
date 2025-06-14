import os
import json
import torch
from kiwipiepy import Kiwi
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def save_json(fileSavePath, saveFile):
    os.makedirs('result', exist_ok=True)
    with open(fileSavePath, "w", encoding="utf-8") as f:
        json.dump(saveFile, f, ensure_ascii=False, indent=2)
    print(f"{fileSavePath} file 생성 완료")

def load_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_tokenizer():
    # https://huggingface.co/nlp04/korean_sentiment_analysis_kcelectra
    tokenizer = AutoTokenizer.from_pretrained("nlp04/korean_sentiment_analysis_kcelectra")
    return tokenizer

def load_model():
    # https://huggingface.co/nlp04/korean_sentiment_analysis_kcelectra
    model = AutoModelForSequenceClassification.from_pretrained("nlp04/korean_sentiment_analysis_kcelectra")
    return model

def get_sentiment_map():
    id2label = {
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
        sentiment_map[i] = decoded.encode('latin1').decode('utf-8')

    return sentiment_map

def load_kiwi():
    kiwi = Kiwi(typos='basic')
    return kiwi

def load_typo_dict():
    with open("assets/typo_dict.json", "r", encoding="utf-8") as f:
        typo_dict = json.load(f)
    return typo_dict