"""두 출처의 JSONL을 동일 Schema로 불러오는 Dataset.
Conversation+Final Request를 ModernBERT로 Tokenization.
Failure/State Label 및 Missing Label Mask 생성.
Gold State와 Target Prediction은 모델 입력에서 제외.
"""
