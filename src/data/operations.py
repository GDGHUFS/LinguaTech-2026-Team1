"""대화 단위 operation 학습 label을 만든다. 추론 입력에는 사용하지 않는다."""
OPERATION_CLASSES = ('ADD', 'MODIFY', 'REVOKE', 'EXCEPTION', 'REINSTATE', 'NO_CHANGE')


def operation_presence(trace):
    """OPERATION_CLASSES 순서의 6차원 multi-hot label을 반환한다.

    UPDATE는 MODIFY로 합친다. 실제 변경 연산이 하나라도 있으면 NO_CHANGE는 0이며, 변경 연산이 없거나 trace가 비어 있으면 1이다."""
    operations = {event.get('op', event.get('operation')) for event in trace}
    operations = {'MODIFY' if op == 'UPDATE' else op for op in operations}
    if operations - set(OPERATION_CLASSES):
        raise ValueError('Unknown policy operation')
    operations.discard('NO_CHANGE')
    if not operations:
        operations.add('NO_CHANGE')
    return [int(op in operations) for op in OPERATION_CLASSES]
