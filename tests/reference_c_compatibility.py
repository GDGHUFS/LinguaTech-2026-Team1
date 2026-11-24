"""보존된 C 브랜치 테스트를 현재 코드에 명시적으로 실행한다.

일부 기대값은 이전 의미론을 따르므로 기본 수집 대상에서 제외한다. 별도 실행 시 실패를 그대로 보고하며 xfail로 숨기지 않는다. 브랜치 전환이나 소스·실험 데이터 수정은 하지 않는다."""
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRANCH = 'hge-1113-b-tests'
FILES = ('conftest.py','test_contracts.py','test_dataset_predictor.py','test_labeling.py',
         'test_mutations_validators.py','test_splitting.py','test_train_utils.py')

for filename in FILES:
    source = subprocess.run(['git','show',f'{BRANCH}:tests/{filename}'],cwd=ROOT,
        capture_output=True,check=True).stdout.decode('utf-8')
    module = types.ModuleType('_reference_c_' + filename[:-3])
    module.__file__ = str(ROOT/'tests'/filename)
    sys.modules[module.__name__] = module
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    for name,value in vars(module).items():
        if name.startswith('test_'):
            globals()['test_reference_c_'+name[5:]]=value
        elif filename=='conftest.py' and name=='seed':
            seed=value
