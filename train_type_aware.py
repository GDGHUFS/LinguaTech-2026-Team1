"""공통 학습 진입점으로 B_pred와 B_oracle을 실행한다.

B_pred는 텍스트에서 예측한 type을 사용한다. gold type을 쓰는 B_oracle은 일반 추론과 분리된 진단 실험이다."""

if __name__ == '__main__':
    # B의 공통 학습 진입점을 재사용한다.
    from train import main
    main()
