"""학습된 예측 콜백과 명시적으로 초기화한 Target으로 데모를 실행한다.

최종 결과만 반환하며 실험 데이터와 독립적으로 동작한다. 학습 데이터 저장이나 자동 API 초기화는 하지 않는다."""
from src.data.oracle_adapter import annotate_dialogue
from src.data.labeling import run_target,failure_labels
from src.evaluation.metrics import probability


def run_demo(scenario,predictor,target,template):
    if scenario.get('demo_only') is not True or scenario.get('data_kind') != 'mock':
        raise ValueError('Demo scenarios must be explicitly demo_only and mock; never research data')
    if scenario.get('language','en') != 'en':
        raise ValueError('Unvalidated demo language/distribution; separate demo-only protocol required')
    if scenario.get('split') is not None:
        raise ValueError('Demo scenarios cannot enter train/validation/test splits')
    dialogue=scenario['dialogue']
    annotation=annotate_dialogue(scenario,dialogue)
    risk=predictor(dialogue['turns'][:-1],dialogue['final_request'])
    score=probability(risk['failure_probability'])
    record=dict(scenario,**{k:annotation[k] for k in ('gold_decision','gold_state','annotation_status')})
    record.update(turns=dialogue['turns'],final_request=dialogue['final_request'])
    if target is None:
        outcome=dict(parse_status='NOT_RUN',target_decision=None,**failure_labels(record['gold_decision'],None))
    else:
        outcome=run_target(record,target,template)
    return dict(mode='demo_only_not_research',gold_state=annotation['gold_state'],gold_decision=annotation['gold_decision'],
        failure_probability=score,target_decision=outcome['target_decision'],parse_status=outcome['parse_status'],
        **{k:outcome[k] for k in ('failure_label_unsafe','failure_label_any','premature_allow')})
