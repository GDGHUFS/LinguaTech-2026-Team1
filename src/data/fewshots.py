"""LLM 호출 없이 합성 생성 예시의 구조를 검증한다."""
import json
import re

from src.data.validators import validate_dialogue
from src.data.oracle_adapter import annotate_dialogue

HELDOUT_PREFIX = 'what would happen if'


def normalized_request(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def uses_heldout_expression(text):
    return HELDOUT_PREFIX in normalized_request(text)


def load_fewshots(path, heldout_templates=()):
    """고정된 합성 예시 7개의 구조, oracle 일치, 연산 범위와 알려진 holdout 표현 누수를 검사한다. 위반 시 ValueError를 발생시킨다."""
    examples = json.loads(path.read_text(encoding='utf-8-sig'))
    shots = examples.get('shots') if isinstance(examples, dict) else None
    if not isinstance(shots, list) or len(shots) != 7:
        raise ValueError('Naturalization requires exactly seven fixed synthetic examples')
    families = [s['template_family_id'] for s in shots]
    if (any(not isinstance(f, str) or not f.strip() for f in families) or
            len(set(families)) != 7 or set(families) & set(heldout_templates)):
        raise ValueError('Duplicate or held-out few-shot template family')
    operations = set()
    for shot in shots:
        sample, output = shot['input'], shot['output']
        from src.rules.schema import validate_trace_priorities
        validate_trace_priorities(sample['initial_rules'], sample['operation_trace'])
        for operation in sample['operation_trace']:
            if not {'operation_id', 'operation'} <= operation.keys() or {'event_id', 'op'} & operation.keys():
                raise ValueError('Few-shot trace requires canonical generation operation_id/operation fields')
            operations.add(operation['operation'])
        if validate_dialogue(output, sample):
            raise ValueError('Invalid few-shot dialogue/mappings')
        if any(uses_heldout_expression(t['text']) for t in output['turns']):
            raise ValueError('Held-out expression leaked into few-shot examples')
        # 기존 adapter로 operation을 op/event_id로 변환해 oracle 결과를 비교한다. 자연어 의미 검증은 별도다.
        annotate_dialogue(sample, output)
    if operations != {'ADD', 'MODIFY', 'REVOKE', 'EXCEPTION', 'REINSTATE', 'NO_CHANGE'}:
        raise ValueError('Few-shot operation coverage must preserve all six operations')
    return examples


def check_fewshot_overlap(rows, shots, heldout_templates):
    requests = {normalized_request(s['output']['final_request']) for s in shots}
    for row in rows:
        if row.get('template_family_id') in heldout_templates:
            if normalized_request(row.get('final_request') or '') in requests:
                raise ValueError('OOD final request copied from few-shot examples')


def expression_audit(rows):
    """표현의 빈도만 집계한다. 의미적 다양성의 통과 기준은 적용하지 않는다."""
    result = {}
    for cap in ('CAP07', 'CAP08'):
        texts = [normalized_request(r['final_request']) for r in rows
                 if r.get('capability_id') == cap and r.get('final_request')]
        result[cap] = dict(generated_count=len(texts), distinct_final_requests=len(set(texts)),
                           semantic_diversity_status='NOT_MEASURED_REQUIRES_QWEN_AND_HUMAN_REVIEW')
    return result
