"""--help만 호출해 CLI 진입점을 검사한다. 실험 명령은 실행하지 않는다."""
import subprocess
import os
import sys
import pytest


@pytest.mark.parametrize('entry',[
    ['scripts/01_data_setup.py'],['scripts/02_prepare.py'],['scripts/03_label_and_split.py'],
    ['scripts/04_export_target_pregen.py'],['scripts/04_pilot_report.py'],['scripts/05_preflight.py'],
    ['scripts/train_baseline_p2.py'],['train.py'],['train_type_aware.py'],['export_predictions.py'],['evaluate.py'],
    ['-m','src.evaluation.baselines_pre'],['-m','src.evaluation.baselines_target_probe'],
    ['-m','src.evaluation.representation_probe'],['-m','src.evaluation.optional']])
def test_cli_help_without_api_or_model_download(entry):
    if entry[0] in ('train.py','train_type_aware.py'):
        pytest.importorskip('torch',reason='training CLI requires declared model runtime')
    environment=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    result=subprocess.run([sys.executable,*entry,'--help'],capture_output=True,text=True,timeout=60,env=environment)
    assert result.returncode==0,result.stderr
    assert 'usage:' in result.stdout.lower()
