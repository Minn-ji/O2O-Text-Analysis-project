### O2O Text Analysis Project

**💡 O2O (Online to Offline) Service**
> 온라인에서 시작된 사용자 경험이 오프라인에서 종료되는 서비스 방식을 의미
---
### (1) Project Summary 📌

O2O 서비스 플랫폼에서 수집된 사용자 리뷰 데이터를 기반으로,  
1.  감정 분석 (model based)
  [`nlp04/korean-sentiment-analysis-kcelectra`](https://huggingface.co/nlp04/korean_sentiment_analysis_kcelectra)  
  > - Hugging Face에 공개된 한국어 감정 분류 모델  
  > - KcELECTRA 기반, 총 11가지 감정 라벨 분류 지원  
2.  감성 분석 (자체 사전 구축)
3. LDA 기반 토픽 모델링  
4. 불만 키워드 분석(WordCloud)

을 통해 사용자 경험을 정량화하고, 서비스 개선에 필요한 인사이트를 도출하는 프로젝트입니다.

---

### (2) Platforms Used for Data Collection 📦

| 서비스 영역     | 플랫폼 명           |
|----------------|--------------------|
| 배달           | 요기요 (Yogiyo)     |
| 숙박           | NOL, 여기어때       |
| 이동 & 교통     | 카카오택시, 우버택시 |
| 항공 & 여행     | 스카이스캐너 (Skyscanner) |

---

### (3) Main functions
- 리뷰 전처리 및 통합: 플랫폼별 리뷰 데이터를 형태소 분석 및 품사 필터링하여 통합
- 감정 분석: Electra 기반 모델 or 감성 사전 구축을 기반으로 긍정/중립/부정 감정 예측 및 정확도 평가
- 토픽 모델링: 리뷰 데이터를 기반으로 LDA 모델 학습 및 주요 불만 유형 도출
- 분포 분석: 앱별 감정 분포 및 토픽 비중 시각화

---
### (4) Directory structure
```
O2O-Text-Analysis-project/
├── assets/
│   ├── malgun.ttf
│   └── typo_dict.json
├── CrawlReviews/
│   ├── crawl_with_scraper.py
│   └── crawl_with_selenium.py
├── SentimentAnalysis/
│   ├── sentiment_dictionary_based.py
│   ├── sentiment_model_based.py
│   └── sentiment_result.ipynb
├── TopicModeling/
│   ├── topic_modeling.py
│   └── topic_modeling_notebook.ipynb
├── WordCloud/
│   ├── draw_wordCloud_sw.py
│   └── draw_wordCloud.py
├── DrawPlot/
│   ├── plot_average_score_over_time.py
│   ├── ...
│   └── plot_cumulative_reviews.py
├── util/
│   ├── basic_tools.py
│   ├── temporal_review_util.py
│   └── text_preprocessing_util.py
├── runner.py
├── requirements.txt
└── README.md
```

---

### (5) How to Run 🚀

**1. Clone repository and install required packages**
```bash
git clone https://github.com/Minn-ji/O2O-Text-Analysis-project.git
cd O2O-Text-Analysis-project
pip install -r requirements.txt
```
**2. Run**
```bash
python -m runner
```
