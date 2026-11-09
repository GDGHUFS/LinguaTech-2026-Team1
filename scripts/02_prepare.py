"""[A] 두 데이터셋을 공통 Disclosure Scenario로 변환.
Python으로 3~9턴 Transition Sequence 생성, Qwen Few-shot Prompt로 자연어 멀티턴 대화 생성.
반복 변경, 부분 예외, 무관한 Turn 등 난도 다양화.
prompts/generation.txt와 prompts/validation.txt를 사용해 구조-자연어 일치 여부 검증 및 실패 사례 분리.

Rule Engine으로 각 생성 대화의 Gold State 계산.
최종 공개 요청에 대한 Gold Decision과 근거 저장.
규칙 충돌/권한/예외 검증 및 모호한 Label 제외.
"""
