"""ModernBERT Shared Encoder 기반 Failure Predictor.
Failure Head(Binary) 및 State Heads(Multi-field) 구현.
A는 Failure Loss만, C는 Failure+State Loss로 학습.
State Head 출력은 Failure Head 입력에 직접 연결하지 않음.
"""
