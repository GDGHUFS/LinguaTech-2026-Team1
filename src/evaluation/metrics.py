"""저장된 점수로 평가 지표를 계산한다. 모델 의존성이나 테스트 시점 학습은 없다.

AUPRC는 B의 검증 지표와 같은 동점 묶음 방식의 average precision이다. 정의할 수 없는 지표는 사유와 함께 null로 반환한다."""
import itertools
import math


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Missing, nonfinite, or out-of-range probability')
    return float(value)


def _pairs(labels, scores):
    if len(labels) != len(scores):
        raise ValueError('Label/score length mismatch')
    if any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError('Observed binary labels required; mask before calling')
    return [(y, probability(p)) for y, p in zip(labels, scores)]


def average_precision(labels, scores):
    """동점 점수를 묶어 average precision을 구한다. 양성이나 음성이 없으면 None을 반환한다."""
    pairs = _pairs(labels, scores)
    if {y for y, _ in pairs} != {0, 1}:
        return None
    positives = sum(labels)
    tp = n = 0
    ap = 0.
    for _, group in itertools.groupby(sorted(pairs, key=lambda x: -x[1]), key=lambda x: x[1]):
        group = list(group)
        added = sum(y for y, _ in group)
        tp += added
        n += len(group)
        # B 학습기와 동점 처리 및 부동소수점 연산 순서를 맞춘다.
        ap += (added / positives) * (tp / n)
    return ap


def precision_recall(labels, scores, threshold):
    pairs = _pairs(labels, scores)
    threshold = probability(threshold)
    tp = sum(y == 1 and p >= threshold for y, p in pairs)
    fp = sum(y == 0 and p >= threshold for y, p in pairs)
    fn = sum(y == 1 and p < threshold for y, p in pairs)
    return {'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None, 'tp': tp, 'fp': fp, 'fn': fn}


def binary_metrics(labels, scores, *, threshold=None, bins=10):
    pairs = _pairs(labels, scores)
    if type(bins) is not int or bins < 1:
        raise ValueError('Positive integer ECE bin count required')
    n, positives = len(pairs), sum(labels)
    result = dict(status='OK', n=n, positives=positives, prevalence=positives/n if n else None,
                  auprc=None, auroc=None, brier=None, ece=None, precision=None, recall=None,
                  threshold=threshold, reasons=[], reliability=[], pr_curve=[])
    if not n:
        result.update(status='NA', reasons=['empty_subset'])
        return result
    result['brier'] = sum((p-y)**2 for y, p in pairs)/n
    reliability = []
    for i in range(bins):
        bucket = [(y, p) for y, p in pairs if min(int(p*bins), bins-1) == i]
        if bucket:
            reliability.append(dict(bin=i, n=len(bucket), predicted=sum(p for _, p in bucket)/len(bucket),
                                    observed=sum(y for y, _ in bucket)/len(bucket)))
    result['reliability'] = reliability
    result['ece'] = sum(b['n']/n*abs(b['predicted']-b['observed']) for b in reliability)
    if {y for y, _ in pairs} == {0, 1}:
        result['auprc'] = average_precision(labels, scores)
        neg_seen = u = 0.
        for _, group in itertools.groupby(sorted(pairs, key=lambda x: x[1]), key=lambda x: x[1]):
            group = list(group)
            pos = sum(y for y, _ in group)
            neg = len(group)-pos
            u += pos*(neg_seen+.5*neg)
            neg_seen += neg
        result['auroc'] = u/(positives*(n-positives))
        tp = count = 0
        for score, group in itertools.groupby(sorted(pairs, key=lambda x: -x[1]), key=lambda x: x[1]):
            group = list(group)
            tp += sum(y for y, _ in group)
            count += len(group)
            result['pr_curve'].append(dict(threshold=score, precision=tp/count, recall=tp/positives))
    else:
        result['reasons'].append('single_class_ranking_undefined')
    if threshold is None:
        result['reasons'].append('threshold_unavailable')
    else:
        result.update(precision_recall(labels, scores, threshold))
        if result['precision'] is None:
            result['reasons'].append('no_predicted_positive')
        if result['recall'] is None:
            result['reasons'].append('no_observed_positive')
    if result['reasons']:
        result['status'] = 'PARTIAL'
    return result


def select_threshold(labels, scores):
    """검증 데이터에서 F1이 최대인 임계값을 고른다. 동률이면 더 높은 임계값을 선택한다."""
    pairs = _pairs(labels, scores)
    if {y for y, _ in pairs} != {0, 1}:
        return {'status': 'NA', 'threshold': None, 'reason': 'validation_requires_two_classes'}
    best = None
    for threshold in sorted({p for _, p in pairs}):
        pr = precision_recall(labels, scores, threshold)
        f1 = 2*pr['tp']/(2*pr['tp']+pr['fp']+pr['fn'])
        if best is None or (f1, threshold) > (best[0], best[1]):
            best = (f1, threshold)
    return {'status': 'OK', 'threshold': best[1], 'validation_f1': best[0]}


def classification_metrics(labels, predictions, classes):
    """관측된 label만 평가한다. macro F1은 선언한 전체 class를, balanced accuracy는 관측된 class를 기준으로 계산한다."""
    if len(labels) != len(predictions):
        raise ValueError('Classification length mismatch')
    observed = [(y, p) for y, p in zip(labels, predictions) if y is not None and y != -100]
    if any(y not in classes or p not in classes for y, p in observed):
        raise ValueError('Missing or unknown class prediction')
    if not observed:
        return dict(status='NA', reason='no_observed_labels', n=0, accuracy=None, macro_f1=None, balanced_accuracy=None)
    f1s, recalls = [], []
    for cls in classes:
        tp = sum(y == p == cls for y, p in observed)
        fp = sum(y != cls and p == cls for y, p in observed)
        fn = sum(y == cls and p != cls for y, p in observed)
        f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        if tp+fn:
            recalls.append(tp/(tp+fn))
    return dict(status='OK', n=len(observed), coverage=len(observed)/len(labels),
                accuracy=sum(y == p for y, p in observed)/len(observed),
                macro_f1=sum(f1s)/len(f1s), balanced_accuracy=sum(recalls)/len(recalls),
                class_counts={str(c): sum(y == c for y, _ in observed) for c in classes})
