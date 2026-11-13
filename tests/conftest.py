"""[B] 테스트 공통 설정. 저장소 루트를 import 경로에 추가하고 가상 seed를 만든다.

여기서 만드는 값은 테스트용 가상 입력이며 data/ 아래에 저장하지 않는다."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def seed():
    return {
        "status": "eligible_pending_review",
        "seed_id": "CoPriva:seed-1",
        "source_seed_id": "seed-1",
        "source_dataset": "CoPriva",
        "source_record_id": "rec-1",
        "group_id": "CoPriva:seed-1",
        "source_provenance": {"repository": "test", "path": "test.json"},
        "policy_context": {"note": "test"},
        "data_type": "medical record",
        "subject": "patient",
        "sender": "doctor",
        "recipient": "insurer",
        "purpose": "claim review",
    }
