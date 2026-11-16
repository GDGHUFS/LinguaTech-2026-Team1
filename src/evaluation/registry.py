"""평가와 실험 진입점에서 공유하는 모델 등록 정보를 검증한다."""
from pathlib import Path
import yaml
from src.utils.reproducibility import content_hash


def load_registry(path='configs/model_registry.yaml'):
    value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    required = {'G1','P1','A','B_pred','C','C+'}
    if {name for name,row in value['models'].items() if row['eval_inclusion'] == 'fair'} != required:
        raise ValueError('Registry must include exactly G1/P1/A/B_pred/C/C+ as required fair models')
    if any(value['models'][name]['eval_inclusion'] == 'fair' for name in ('B_oracle','Heo')):
        raise ValueError('Oracle and post-run diagnostics cannot enter fair inference comparison')
    return value, content_hash(value)
