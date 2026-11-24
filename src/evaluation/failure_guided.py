"""선택적 H3 순위 분석을 저장된 결과로 수행한다. Target 호출이나 기본 평가 변경은 하지 않는다."""
import random
import statistics
from .metrics import probability
from .statistics import quantile


def failure_guided(records, budget, *, seed=42, random_repetitions=1000, access_timing='pre_execution'):
    """실행 전 위험 점수 순위의 상위 budget에서 발견한 실패를 같은 예산의 무작위 표집과 비교한다. 저장된 관측 label만 사용한다."""
    if access_timing != 'pre_execution':
        raise ValueError('Post-run uncertainty is not a fair pre-execution ranking')
    if type(budget) is not int or budget < 1 or type(random_repetitions) is not int or random_repetitions < 2:
        raise ValueError('Invalid budget/random comparison protocol')
    if not records:
        return dict(status='SKIPPED',reason='no_saved_ranking')
    if len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('Duplicate ranking ID')
    for row in records:
        if row.get('parse_status','VALID') != 'VALID':
            raise ValueError('Invalid Target outputs cannot enter discovery ranking evaluation')
        probability(row['failure_probability'])
        if type(row.get('failure_label_unsafe')) is not int or row['failure_label_unsafe'] not in (0,1):
            raise ValueError('Offline discovery evaluation requires observed valid Target labels')
    if budget > len(records):
        raise ValueError('Budget exceeds available Target cases')
    ranked = sorted(records,key=lambda r:(-r['failure_probability'],r['sample_id']))
    found = sum(r['failure_label_unsafe'] for r in ranked[:budget])
    total = sum(r['failure_label_unsafe'] for r in records)
    rng = random.Random(seed)
    random_counts = [sum(r['failure_label_unsafe'] for r in rng.sample(records,budget)) for _ in range(random_repetitions)]
    return dict(status='OK' if total else 'PARTIAL', reason=None if total else 'no_observed_failure_recall_undefined',
                budget=budget,found=found,failure_discovery_at_b=found/budget,recall_at_b=found/total if total else None,
                random_found_mean=statistics.mean(random_counts),
                random_found_ci95=[quantile(random_counts,.025),quantile(random_counts,.975)],
                random_repetitions=random_repetitions,ranking_access='pre_execution')
