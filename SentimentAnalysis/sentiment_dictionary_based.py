import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, \
    recall_score
import warnings
import re
from collections import Counter
from typing import Tuple, Dict, List, Optional
import logging

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['Malgun Gothic', 'Apple Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleBinarySentimentLexicon:

    def __init__(self):
        # 핵심 긍정어 (실제 리뷰에서 자주 사용되는 단어들)
        self.positive_words = {
            # 맛 관련 긍정어
            '맛있': 3.0, '맛나': 2.5, '맛집': 2.8, '꿀맛': 3.2, '존맛': 3.5,
            '개맛있': 3.3, '핵맛있': 3.4, '진짜맛있': 3.0, '대존맛': 3.5,
            '고소': 2.0, '달콤': 2.2, '바삭': 2.0, '촉촉': 2.1, '부드럽': 2.0,
            '신선': 2.3, '깔끔': 2.1, '담백': 1.8, '진한': 1.9, '풍부': 2.0,

            # 서비스 긍정어
            '친절': 2.5, '빠르': 2.3, '신속': 2.4, '깨끗': 2.1, '정성': 2.2,
            '세심': 2.0, '배려': 2.1, '따뜻': 1.9, '미소': 1.8, '정중': 2.0,

            # 일반 긍정어
            '좋': 2.0, '최고': 3.0, '훌륭': 2.8, '완벽': 3.0, '대박': 3.2,
            '짱': 2.8, '갓': 3.0, '킹': 2.9, '쩔': 2.7, '개쩔': 3.0,
            '만족': 2.5, '추천': 2.7, '강추': 3.0, '극추': 3.2,
            '성공': 2.3, '확실': 2.1, '완전': 1.8, '정말': 1.5,

            # 가성비/품질
            '가성비': 2.8, '혜자': 3.0, '존혜': 3.2, '꿀': 2.5, '개꿀': 2.8,
            '합리적': 2.2, '저렴': 2.0, '싸': 1.8, '실속': 2.1,
            '품질': 2.0, '퀄리티': 2.2, '고급': 2.3, '프리미엄': 2.4,

            # 강한 긍정 표현
            '미쳤': 3.5, '미친': 3.3, '돌았': 3.0, '장난아니': 3.2,
            '실화': 2.8, '레전드': 3.4, '갓작': 3.5, '명작': 3.0,
            '역시': 2.0, '당연': 1.8, '확실히': 2.1, '분명': 1.9,

            # 감정 표현
            '행복': 2.5, '기쁘': 2.3, '즐거': 2.2, '신나': 2.4, '감동': 2.8,
            '고마': 2.1, '감사': 2.3, '사랑': 2.6, '힐링': 2.0,
        }

        # 핵심 부정어
        self.negative_words = {
            # 맛 관련 부정어
            '맛없': 3.0, '노맛': 3.2, '헬맛': 3.5, '개맛없': 3.3, '핵맛없': 3.4,
            '쓰레기맛': 3.5, '똥맛': 3.5, '짜': 2.0, '싱거': 1.8, '느끼': 2.1,
            '비릿': 2.3, '상했': 3.0, '썩': 3.2, '역겨': 3.0, '토할': 3.2,
            '딱딱': 2.0, '차갑': 1.8, '이상': 2.2, '구린': 2.5,

            # 서비스 부정어
            '불친절': 3.0, '늦': 2.2, '느림': 2.0, '더럽': 2.5, '지저분': 2.3,
            '무례': 2.8, '싸가지': 3.2, '태도': 2.0, '불결': 2.7, '비위생': 2.8,

            # 일반 부정어
            '별로': 2.5, '나쁘': 2.3, '최악': 3.5, '끔찍': 3.2, '형편없': 3.0,
            '쓰레기': 3.5, '똥': 3.5, '헬': 3.3, '지옥': 3.4, '망': 3.0,
            '실망': 2.8, '후회': 2.5, '짜증': 2.7, '화': 2.4, '환불': 2.9,
            '돈아깝': 3.0, '시간낭비': 3.2, '다신안': 3.5, '비추': 3.0,

            # 품질 부정어
            '조악': 2.8, '허술': 2.5, '불량': 2.9, '결함': 2.7, '하자': 2.6,
            '문제': 2.2, '고장': 2.8, '망가진': 2.9, '엉성': 2.4,

            # 미묘한 부정 표현
            '그냥그래': 1.8, '뭔가아쉬': 2.0, '좀아쉬': 1.9, '미묘': 1.5,
            '애매': 1.6, '글쎄': 1.4, '흠': 1.5, '그럭저럭': 1.3,
            '무난': 1.2, '평범': 1.3, '보통': 1.1, '어중간': 1.7,
            '밋밋': 1.8, '재미없': 2.2, '지루': 2.0, '노잼': 2.5,
        }

        # 강도 조절어 (간소화)
        self.intensifiers = {
            # 강화어
            '정말': 1.4, '진짜': 1.5, '너무': 1.3, '엄청': 1.6, '완전': 1.4,
            '핵': 1.8, '개': 1.6, '존': 1.5, '겁나': 1.6, '무지': 1.5,
            '극': 1.7, '초': 1.3, '슈퍼': 1.4, '매우': 1.3, '아주': 1.3,

            # 약화어
            '좀': 0.8, '조금': 0.7, '약간': 0.8, '살짝': 0.7, '그냥': 0.9,
            '나름': 0.8, '어느정도': 0.8, '제법': 0.9,
        }

        # 부정어
        self.negation_words = {'안', '않', '못', '없', '아니', '말고'}

        # 이모티콘 (간소화)
        self.positive_emoticons = {
            r'[ㅋㅎ]{2,}': 1.5,  # ㅋㅋ, ㅎㅎ
            r'[!]{2,}': 1.2,  # !!
        }

        self.negative_emoticons = {
            r'[ㅠㅜ]{2,}': 1.6,  # ㅠㅠ, ㅜㅜ
            r'[ㅡ]{2,}': 1.1,  # ㅡㅡ
        }


class BinarySentimentAnalyzer:
    """이진 감정 분석기 (긍정/부정)"""

    def __init__(self):
        self.lexicon = SimpleBinarySentimentLexicon()
        self.konlpy_available = self._check_konlpy()

    def _check_konlpy(self) -> bool:
        """KoNLPy 사용 가능 여부 확인"""
        try:
            from konlpy.tag import Okt
            self.tagger = Okt()
            logger.info("KoNLPy Okt 형태소 분석기를 사용합니다.")
            return True
        except ImportError:
            logger.warning("KoNLPy가 설치되지 않아 키워드 기반 분석을 사용합니다.")
            self.tagger = None
            return False

    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if pd.isna(text):
            return ""

        text = str(text).lower()
        # 기본적인 정리만 수행
        text = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣㅋㅎㅠㅜㅡ!]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_emoticon_sentiment(self, text: str) -> Tuple[float, float]:
        """이모티콘 감정 점수 추출"""
        pos_score = 0
        neg_score = 0

        for pattern, score in self.lexicon.positive_emoticons.items():
            matches = len(re.findall(pattern, text))
            pos_score += matches * score

        for pattern, score in self.lexicon.negative_emoticons.items():
            matches = len(re.findall(pattern, text))
            neg_score += matches * score

        return pos_score, neg_score

    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """감정 분석 수행

        Returns:
            (sentiment: 'positive'/'negative', confidence: 0-1)
        """
        if pd.isna(text) or not text.strip():
            return 'neutral', 0.1

        text = self.preprocess_text(text)

        # 이모티콘 점수
        emo_pos, emo_neg = self.extract_emoticon_sentiment(text)

        # 단어 기반 점수 계산
        pos_score = emo_pos
        neg_score = emo_neg

        words = text.split()

        for i, word in enumerate(words):
            # 긍정어 검사
            for pos_word, base_score in self.lexicon.positive_words.items():
                if pos_word in word:
                    score = base_score

                    # 강도 조절어 확인
                    intensifier = self._find_intensifier(words, i)
                    if intensifier > 0:
                        score *= intensifier

                    # 부정어 확인
                    if self._check_negation(words, i):
                        neg_score += score * 0.8  # 부정된 긍정어는 부정으로
                    else:
                        pos_score += score
                    break

            # 부정어 검사
            for neg_word, base_score in self.lexicon.negative_words.items():
                if neg_word in word:
                    score = base_score

                    intensifier = self._find_intensifier(words, i)
                    if intensifier > 0:
                        score *= intensifier

                    if self._check_negation(words, i):
                        pos_score += score * 0.6  # 부정된 부정어는 약한 긍정
                    else:
                        neg_score += score
                    break

        # 최종 판단
        total_score = pos_score + neg_score
        if total_score == 0:
            return 'neutral', 0.1

        pos_ratio = pos_score / total_score

        # 신뢰도 계산
        confidence = min(total_score / 5.0, 1.0)
        confidence = max(confidence, 0.1)

        # 판단 기준을 더 명확하게
        if pos_ratio >= 0.6:  # 60% 이상이면 긍정
            return 'positive', confidence
        elif pos_ratio <= 0.4:  # 40% 이하면 부정
            return 'negative', confidence
        else:
            return 'neutral', confidence * 0.5  # 애매한 경우 신뢰도 낮춤

    def _find_intensifier(self, words: List[str], index: int) -> float:
        """강도 조절어 찾기"""
        for j in range(max(0, index - 2), index):
            word = words[j]
            for intensifier, value in self.lexicon.intensifiers.items():
                if intensifier in word:
                    return value
        return 1.0

    def _check_negation(self, words: List[str], index: int) -> bool:
        """부정어 확인"""
        for j in range(max(0, index - 2), index):
            word = words[j]
            if any(neg in word for neg in self.lexicon.negation_words):
                return True
        return False


class BinarySentimentEvaluator:
    """이진 감정 분석 평가기"""

    @staticmethod
    def convert_score_to_binary(scores: np.ndarray) -> np.ndarray:
        """5점 척도를 이진으로 변환"""
        # 1,2 -> negative, 4,5 -> positive, 3 -> neutral (제외)
        binary_labels = []
        for score in scores:
            if score <= 2:
                binary_labels.append('negative')
            elif score >= 4:
                binary_labels.append('positive')
            else:
                binary_labels.append('neutral')  # 중립은 평가에서 제외
        return np.array(binary_labels)

    @staticmethod
    def evaluate_binary_performance(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """이진 분류 성능 평가"""

        # 중립 제거 (평가에서 제외)
        mask = (y_true != 'neutral') & (y_pred != 'neutral')
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]

        if len(y_true_filtered) == 0:
            return {'error': 'No valid data for evaluation'}

        # 성능 지표 계산
        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)

        # positive를 1, negative를 0으로 변환하여 이진 분류 메트릭 계산
        y_true_binary = (y_true_filtered == 'positive').astype(int)
        y_pred_binary = (y_pred_filtered == 'positive').astype(int)

        precision = precision_score(y_true_binary, y_pred_binary, average='binary')
        recall = recall_score(y_true_binary, y_pred_binary, average='binary')
        f1 = f1_score(y_true_binary, y_pred_binary, average='binary')

        # 클래스별 성능
        pos_mask = y_true_filtered == 'positive'
        neg_mask = y_true_filtered == 'negative'

        pos_accuracy = accuracy_score(y_true_filtered[pos_mask], y_pred_filtered[pos_mask]) if pos_mask.sum() > 0 else 0
        neg_accuracy = accuracy_score(y_true_filtered[neg_mask], y_pred_filtered[neg_mask]) if neg_mask.sum() > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'positive_accuracy': pos_accuracy,
            'negative_accuracy': neg_accuracy,
            'total_samples': len(y_true_filtered),
            'positive_samples': pos_mask.sum(),
            'negative_samples': neg_mask.sum()
        }


class SentimentDistributionAnalyzer:
    """감정 분포 분석기"""

    @staticmethod
    def analyze_binary_distribution(predictions: np.ndarray) -> Dict:
        """이진 분류 분포 분석"""

        total_count = len(predictions)

        # 각 클래스 개수
        positive_count = (predictions == 'positive').sum()
        negative_count = (predictions == 'negative').sum()
        neutral_count = (predictions == 'neutral').sum()

        distribution = {
            'total_reviews': total_count,
            'positive': {'count': positive_count, 'percentage': round(positive_count / total_count * 100, 2)},
            'negative': {'count': negative_count, 'percentage': round(negative_count / total_count * 100, 2)},
            'neutral': {'count': neutral_count, 'percentage': round(neutral_count / total_count * 100, 2)},
        }

        return distribution

    @staticmethod
    def print_distribution_summary(distribution: Dict, title: str = "감정 분포"):
        """분포 요약 출력"""

        print(f"\n감정 분포: {title}")
        print("=" * 50)
        print(f"총 리뷰 수: {distribution['total_reviews']:,}개")
        print()

        for sentiment in ['positive', 'negative', 'neutral']:
            data = distribution[sentiment]
            count = data['count']
            pct = data['percentage']

            if sentiment == 'positive':
                name = '긍정'
                color = '[+]'
            elif sentiment == 'negative':
                name = '부정'
                color = '[-]'
            else:
                name = '중립'
                color = '[=]'

            bar = "█" * int(pct / 2)
            print(f"{color} {name}: {count:6,}개 ({pct:5.1f}%) {bar}")


class BinarySentimentRunner:
    """이진 감정 분석 실행기"""

    def __init__(self, sample_size: int = 5000):
        self.analyzer = BinarySentimentAnalyzer()
        self.evaluator = BinarySentimentEvaluator()
        self.distribution_analyzer = SentimentDistributionAnalyzer()
        self.sample_size = sample_size

    def run_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """분석 실행"""

        logger.info(f"데이터 분석: 총 {len(df):,}개 리뷰에서 {self.sample_size:,}개 샘플 분석 시작")

        # 샘플링
        if len(df) > self.sample_size:
            df_sample = df.sample(n=self.sample_size, random_state=42).reset_index(drop=True)
        else:
            df_sample = df.copy()

        # 원본 데이터 분포 분석 (5점 척도 -> 이진 변환)
        if 'score' in df.columns:
            original_binary = self.evaluator.convert_score_to_binary(df['score'].values)
            original_dist = self.distribution_analyzer.analyze_binary_distribution(original_binary)
            self.distribution_analyzer.print_distribution_summary(original_dist, "원본 데이터 분포 (5점→이진 변환)")

        # 예측 수행
        logger.info("감정 분석 수행 중...")
        predictions = []
        confidences = []

        for text in df_sample['text']:
            sentiment, confidence = self.analyzer.analyze_sentiment(text)
            predictions.append(sentiment)
            confidences.append(confidence)

        df_sample['predicted_sentiment'] = predictions
        df_sample['confidence'] = confidences

        # 예측 분포 분석
        pred_dist = self.distribution_analyzer.analyze_binary_distribution(np.array(predictions))
        self.distribution_analyzer.print_distribution_summary(pred_dist, "예측 결과 분포")

        return df_sample

    def evaluate_and_visualize(self, df: pd.DataFrame) -> Dict:
        """평가 및 시각화"""

        if 'score' not in df.columns:
            logger.error("원본 점수 컬럼이 없어 평가를 수행할 수 없습니다.")
            return {}

        # 원본 점수를 이진으로 변환
        y_true = self.evaluator.convert_score_to_binary(df['score'].values)
        y_pred = df['predicted_sentiment'].values

        # 성능 평가
        performance = self.evaluator.evaluate_binary_performance(y_true, y_pred)

        if 'error' not in performance:
            logger.info("\n성능 평가 결과:")
            logger.info(f"  정확도: {performance['accuracy']:.3f}")
            logger.info(f"  정밀도: {performance['precision']:.3f}")
            logger.info(f"  재현율: {performance['recall']:.3f}")
            logger.info(f"  F1 점수: {performance['f1_score']:.3f}")
            logger.info(f"  긍정 정확도: {performance['positive_accuracy']:.3f}")
            logger.info(f"  부정 정확도: {performance['negative_accuracy']:.3f}")

        # 시각화
        self._create_binary_visualization(df, performance)

        return performance

    def _create_binary_visualization(self, df: pd.DataFrame, performance: Dict):
        """이진 분류 시각화"""

        fig = plt.figure(figsize=(20, 12))

        # 1. 성능 지표 바 차트
        ax1 = plt.subplot(3, 4, 1)
        if 'error' not in performance:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            metric_names = ['정확도', '정밀도', '재현율', 'F1점수']
            values = [performance[metric] for metric in metrics]

            bars = ax1.bar(metric_names, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
            ax1.set_ylabel('점수')
            ax1.set_title('성능 지표')
            ax1.set_ylim(0, 1)

            # 값 표시
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{value:.3f}', ha='center', va='bottom')

        # 2. 원본 분포 (5점 척도)
        ax2 = plt.subplot(3, 4, 2)
        if 'score' in df.columns:
            score_counts = df['score'].value_counts().sort_index()
            colors = ['#ff4444', '#ff8888', '#ffcc88', '#88cc88', '#44aa44']
            bars = ax2.bar(score_counts.index, score_counts.values, color=colors)
            ax2.set_xlabel('점수')
            ax2.set_ylabel('개수')
            ax2.set_title('원본 점수 분포 (5점 척도)')
            ax2.set_xticks(range(1, 6))

        # 3. 이진 변환 분포
        ax3 = plt.subplot(3, 4, 3)
        if 'score' in df.columns:
            binary_true = self.evaluator.convert_score_to_binary(df['score'].values)
            binary_counts = pd.Series(binary_true).value_counts()

            colors = {'positive': '#44aa44', 'negative': '#ff4444', 'neutral': '#ffcc88'}
            bars = ax3.bar(range(len(binary_counts)), binary_counts.values,
                           color=[colors[label] for label in binary_counts.index])
            ax3.set_xlabel('감정')
            ax3.set_ylabel('개수')
            ax3.set_title('이진 변환 분포')
            ax3.set_xticks(range(len(binary_counts)))
            ax3.set_xticklabels(binary_counts.index, rotation=45)

        # 4. 예측 분포
        ax4 = plt.subplot(3, 4, 4)
        pred_counts = df['predicted_sentiment'].value_counts()
        colors = {'positive': '#44aa44', 'negative': '#ff4444', 'neutral': '#ffcc88'}
        bars = ax4.bar(range(len(pred_counts)), pred_counts.values,
                       color=[colors[label] for label in pred_counts.index])
        ax4.set_xlabel('감정')
        ax4.set_ylabel('개수')
        ax4.set_title('예측 결과 분포')
        ax4.set_xticks(range(len(pred_counts)))
        ax4.set_xticklabels(pred_counts.index, rotation=45)

        # 5. 혼동 행렬
        ax5 = plt.subplot(3, 4, 5)
        if 'score' in df.columns and 'error' not in performance:
            y_true = self.evaluator.convert_score_to_binary(df['score'].values)
            y_pred = df['predicted_sentiment'].values

            # 중립 제거
            mask = (y_true != 'neutral') & (y_pred != 'neutral')
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]

            if len(y_true_filtered) > 0:
                cm = confusion_matrix(y_true_filtered, y_pred_filtered,
                                      labels=['negative', 'positive'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                            xticklabels=['Negative', 'Positive'],
                            yticklabels=['Negative', 'Positive'])
                ax5.set_title('혼동 행렬')
                ax5.set_xlabel('예측')
                ax5.set_ylabel('실제')

        # 6. 신뢰도 분포
        ax6 = plt.subplot(3, 4, 6)
        ax6.hist(df['confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax6.set_xlabel('신뢰도')
        ax6.set_ylabel('빈도')
        ax6.set_title('예측 신뢰도 분포')
        ax6.axvline(df['confidence'].mean(), color='red', linestyle='--',
                    label=f'평균: {df["confidence"].mean():.3f}')
        ax6.legend()

        # 7. 감정별 신뢰도
        ax7 = plt.subplot(3, 4, 7)
        sentiment_confidence = df.groupby('predicted_sentiment')['confidence'].mean()
        colors = {'positive': '#44aa44', 'negative': '#ff4444', 'neutral': '#ffcc88'}
        bars = ax7.bar(range(len(sentiment_confidence)), sentiment_confidence.values,
                       color=[colors[label] for label in sentiment_confidence.index])
        ax7.set_xlabel('예측 감정')
        ax7.set_ylabel('평균 신뢰도')
        ax7.set_title('감정별 평균 신뢰도')
        ax7.set_xticks(range(len(sentiment_confidence)))
        ax7.set_xticklabels(sentiment_confidence.index, rotation=45)

        # 값 표시
        for bar, value in zip(bars, sentiment_confidence.values):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{value:.3f}', ha='center', va='bottom')

        # 8. 점수별 예측 정확도 (5점 척도 기준)
        ax8 = plt.subplot(3, 4, 8)
        if 'score' in df.columns:
            score_accuracy = {}
            for score in range(1, 6):
                mask = df['score'] == score
                if mask.sum() > 0:
                    true_binary = self.evaluator.convert_score_to_binary(df.loc[mask, 'score'].values)
                    pred_binary = df.loc[mask, 'predicted_sentiment'].values

                    # 중립 제외하고 정확도 계산
                    eval_mask = (true_binary != 'neutral') & (pred_binary != 'neutral')
                    if eval_mask.sum() > 0:
                        acc = accuracy_score(true_binary[eval_mask], pred_binary[eval_mask])
                        score_accuracy[score] = acc

            if score_accuracy:
                scores = list(score_accuracy.keys())
                accuracies = list(score_accuracy.values())
                ax8.plot(scores, accuracies, marker='o', linewidth=2, markersize=8)
                ax8.set_xlabel('원본 점수')
                ax8.set_ylabel('예측 정확도')
                ax8.set_title('점수별 예측 정확도')
                ax8.set_xticks(range(1, 6))
                ax8.grid(True, alpha=0.3)

        # 9. 신뢰도별 정확도
        ax9 = plt.subplot(3, 4, 9)
        if 'score' in df.columns and 'error' not in performance:
            y_true = self.evaluator.convert_score_to_binary(df['score'].values)
            y_pred = df['predicted_sentiment'].values

            # 중립 제거
            mask = (y_true != 'neutral') & (y_pred != 'neutral')
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            confidence_filtered = df['confidence'].values[mask]

            if len(y_true_filtered) > 0:
                # 신뢰도를 구간별로 나누어 정확도 계산
                confidence_bins = np.linspace(0, 1, 11)
                bin_centers = []
                accuracies = []

                for i in range(len(confidence_bins) - 1):
                    bin_mask = (confidence_filtered >= confidence_bins[i]) & (
                                confidence_filtered < confidence_bins[i + 1])
                    if bin_mask.sum() > 0:
                        acc = accuracy_score(y_true_filtered[bin_mask], y_pred_filtered[bin_mask])
                        bin_centers.append((confidence_bins[i] + confidence_bins[i + 1]) / 2)
                        accuracies.append(acc)

                ax9.plot(bin_centers, accuracies, marker='o', linewidth=2)
                ax9.set_xlabel('신뢰도')
                ax9.set_ylabel('정확도')
                ax9.set_title('신뢰도별 정확도')
                ax9.grid(True, alpha=0.3)

        # 10. 긍정/부정 단어 빈도 (상위 10개)
        ax10 = plt.subplot(3, 4, 10)
        positive_words = []
        negative_words = []

        for text in df['text'].dropna():
            text = self.analyzer.preprocess_text(text)
            words = text.split()

            for word in words:
                for pos_word in self.analyzer.lexicon.positive_words:
                    if pos_word in word:
                        positive_words.append(pos_word)
                        break

                for neg_word in self.analyzer.lexicon.negative_words:
                    if neg_word in word:
                        negative_words.append(neg_word)
                        break

        # 상위 5개씩
        pos_counter = Counter(positive_words).most_common(5)
        neg_counter = Counter(negative_words).most_common(5)

        if pos_counter or neg_counter:
            all_words = []
            all_counts = []
            all_colors = []

            for word, count in pos_counter:
                all_words.append(f'+{word}')
                all_counts.append(count)
                all_colors.append('#44aa44')

            for word, count in neg_counter:
                all_words.append(f'-{word}')
                all_counts.append(count)
                all_colors.append('#ff4444')

            if all_words:
                bars = ax10.barh(range(len(all_words)), all_counts, color=all_colors)
                ax10.set_yticks(range(len(all_words)))
                ax10.set_yticklabels(all_words)
                ax10.set_xlabel('빈도')
                ax10.set_title('주요 감정 단어 (상위 10개)')

        # 11. 예측 정확성 히트맵
        ax11 = plt.subplot(3, 4, 11)
        if 'score' in df.columns and 'error' not in performance:
            # 5점 척도별 예측 분포
            prediction_matrix = np.zeros((5, 3))  # 5점 x 3감정
            sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}

            for score in range(1, 6):
                mask = df['score'] == score
                if mask.sum() > 0:
                    pred_counts = df.loc[mask, 'predicted_sentiment'].value_counts()
                    total = mask.sum()

                    for sentiment, count in pred_counts.items():
                        prediction_matrix[score - 1, sentiment_map[sentiment]] = count / total * 100

            sns.heatmap(prediction_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                        xticklabels=['Negative', 'Neutral', 'Positive'],
                        yticklabels=['1점', '2점', '3점', '4점', '5점'], ax=ax11)
            ax11.set_title('점수별 예측 분포 (%)')
            ax11.set_xlabel('예측 감정')
            ax11.set_ylabel('원본 점수')

        # 12. 전체 요약 정보
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')

        summary_text = f"""
이진 감정분석 요약

데이터 정보:
• 전체 리뷰: {len(df):,}개
• 평균 신뢰도: {df['confidence'].mean():.3f}

성능 지표:"""

        if 'error' not in performance:
            summary_text += f"""
• 정확도: {performance['accuracy']:.3f}
• 정밀도: {performance['precision']:.3f}
• 재현율: {performance['recall']:.3f}
• F1 점수: {performance['f1_score']:.3f}

분포:
• 긍정: {performance['positive_samples']:,}개
• 부정: {performance['negative_samples']:,}개
• 평가 샘플: {performance['total_samples']:,}개"""

        # 어휘 사전 정보
        lexicon = self.analyzer.lexicon
        summary_text += f"""

어휘 사전:
• 긍정어: {len(lexicon.positive_words):,}개
• 부정어: {len(lexicon.negative_words):,}개
• 강도조절어: {len(lexicon.intensifiers):,}개"""

        ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes,
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

        plt.tight_layout()
        plt.savefig('binary_sentiment_analysis.png', dpi=300, bbox_inches='tight')
        logger.info("시각화 결과가 'binary_sentiment_analysis.png'에 저장되었습니다.")

        return fig


def main():
    """메인 실행 함수"""

    print("이진 한국어 감정분석 시스템 시작")
    print("=" * 60)

    # 설정
    file_path = '요기요_reviews_korean_only.csv'
    sample_size = 3000

    try:
        # 데이터 로드
        logger.info("데이터 로드 중...")
        df = pd.read_csv(file_path)
        logger.info(f"총 {len(df):,}개의 리뷰를 로드했습니다.")

        # 어휘 사전 정보 출력
        analyzer = BinarySentimentAnalyzer()
        lexicon = analyzer.lexicon

        print(f"\n어휘 사전 정보:")
        print(f"  긍정어: {len(lexicon.positive_words):,}개")
        print(f"  부정어: {len(lexicon.negative_words):,}개")
        print(f"  강도조절어: {len(lexicon.intensifiers):,}개")
        print(f"  부정접속어: {len(lexicon.negation_words):,}개")
        print(f"  이모티콘 패턴: {len(lexicon.positive_emoticons) + len(lexicon.negative_emoticons):,}개")

        total_words = len(lexicon.positive_words) + len(lexicon.negative_words)
        print(f"  총 감정어휘: {total_words:,}개")

        # 샘플 어휘 출력
        print(f"\n주요 긍정어 예시:")
        pos_samples = list(lexicon.positive_words.items())[:10]
        for word, score in pos_samples:
            print(f"    • '{word}': {score}")

        print(f"\n주요 부정어 예시:")
        neg_samples = list(lexicon.negative_words.items())[:10]
        for word, score in neg_samples:
            print(f"    • '{word}': {score}")

        # 분석 실행
        runner = BinarySentimentRunner(sample_size=sample_size)
        df_analyzed = runner.run_analysis(df)

        # 평가 및 시각화
        performance = runner.evaluate_and_visualize(df_analyzed)

        # 결과 저장
        output_file = 'binary_sentiment_results.csv'
        df_analyzed.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"분석 결과가 '{output_file}'에 저장되었습니다.")

        # 최종 요약
        print("\n" + "=" * 60)
        print("최종 결과 요약")
        print("=" * 60)

        if performance and 'error' not in performance:
            print(f"전체 성능:")
            print(f"  • 정확도: {performance['accuracy']:.1%}")
            print(f"  • 정밀도: {performance['precision']:.1%}")
            print(f"  • 재현율: {performance['recall']:.1%}")
            print(f"  • F1 점수: {performance['f1_score']:.1%}")

            print(f"\n클래스별 성능:")
            print(f"  • 긍정 분류 정확도: {performance['positive_accuracy']:.1%}")
            print(f"  • 부정 분류 정확도: {performance['negative_accuracy']:.1%}")

            print(f"\n데이터 분포:")
            print(f"  • 평가 샘플: {performance['total_samples']:,}개")
            print(f"  • 긍정 샘플: {performance['positive_samples']:,}개")
            print(f"  • 부정 샘플: {performance['negative_samples']:,}개")

        # 주요 개선사항
        print(f"\n주요 특징:")
        print(f"  단순하고 명확한 이진 분류 (긍정/부정)")
        print(f"  실용적인 어휘 사전 ({total_words:,}개 핵심 단어)")
        print(f"  미묘한 부정 표현 처리")
        print(f"  이모티콘 기반 감정 인식")
        print(f"  신뢰도 기반 예측")
        print(f"  종합적인 성능 평가")

        print(f"\n출력 파일:")
        print(f"  시각화: binary_sentiment_analysis.png")
        print(f"  결과 데이터: {output_file}")
        print(f"  분석 리뷰 수: {len(df_analyzed):,}개")

        # 샘플 예측 결과 출력
        print(f"\n샘플 예측 결과:")
        sample_results = df_analyzed.head(5)[['text', 'predicted_sentiment', 'confidence']]
        for idx, row in sample_results.iterrows():
            text = row['text'][:50] + '...' if len(row['text']) > 50 else row['text']
            sentiment = row['predicted_sentiment']
            confidence = row['confidence']

            symbol = '[+]' if sentiment == 'positive' else '[-]' if sentiment == 'negative' else '[=]'
            print(f"  {symbol} \"{text}\" → {sentiment} ({confidence:.2f})")

        plt.show()

    except FileNotFoundError:
        logger.error(f"파일을 찾을 수 없습니다: {file_path}")
        print(f"\n해결 방법:")
        print(f"  1. 파일 경로 확인: {file_path}")
        print(f"  2. 현재 디렉토리에 파일이 있는지 확인")
        print(f"  3. 파일명이 정확한지 확인")
    except Exception as e:
        logger.error(f"오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()