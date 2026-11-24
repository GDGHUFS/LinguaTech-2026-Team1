"""저장된 모델의 점수와 표현을 추출한다. torch는 호출 시 불러온다.

일반 추론에는 gold를 전달하지 않는다. B_oracle의 gold type 입력은 명시적인 진단 모드에서만 허용한다."""
from .metrics import probability


def predict_records(model, tokenizer, records, *, model_name, seed, max_length=2048, checkpoint_hash=None, diagnostic_only=False):
    """record별 점수와 선택적 CLS 표현을 순서대로 반환한다.

    모델을 eval 상태로 바꾸고 모든 parameter의 gradient를 비활성화한다. 이 상태는 복원하지 않는다. 일반 추론 입력은 대화와 요청으로 제한하며 gold type은 명시적인 B_oracle 진단에서만 전달한다."""
    import torch
    if model_name != model.variant:
        raise ValueError('Export model registry name and architecture disagree')
    if (model_name == 'B_oracle' or model.variant == 'B_oracle') and not diagnostic_only:
        raise ValueError('Oracle diagnostic cannot use the fair inference exporter')
    if diagnostic_only and not (model_name == model.variant == 'B_oracle'):
        raise ValueError('Diagnostic gold features require a B_oracle checkpoint')
    from src.models.dataset import DialogueDataset, DialogueCollator, STATE_LABELS
    device = next(model.parameters()).device
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    seen = set()
    for record in records:
        sid = record['sample_id']
        if sid in seen:
            raise ValueError('Duplicate export sample ID')
        seen.add(sid)
        if record.get('request_turn_id') is None:
            raise ValueError('Export requires an explicit request turn ID')
        clean = {k:record[k] for k in ('turns','request_turn_id')}
        clean['final_request'] = record.get('final_request') or (record.get('request') or {}).get('text')
        item = DialogueDataset([clean],tokenizer,max_length)[0]
        if not item['request_mask']:
            raise ValueError('Final request truncated; use the common registered length policy')
        batch = DialogueCollator(tokenizer)([item])
        features = {k:v.to(device) for k,v in batch.items() if k in
                    ('input_ids','attention_mask','turn_spans','turn_mask','request_span','request_mask')}
        if diagnostic_only:
            from src.data.operations import operation_presence
            features['oracle_type_features'] = torch.tensor([operation_presence(record['operation_trace'])],device=device,dtype=torch.float)
        with torch.inference_mode():
            output = model(**features)
        result = dict(sample_id=sid,split=record['split'],group_id=record['group_id'],model=model_name,seed=seed,
                      checkpoint_hash=checkpoint_hash, failure_probability=probability(float(output['failure_logits'].sigmoid()[0])),
                      truncated=item['diagnostics']['truncated'],information_access='gold_type_diagnostic' if diagnostic_only else 'text_pre_execution')
        state_logits = output.get('final_state_logits',output['state_logits'])
        if state_logits:
            result['predicted_state'] = {name:STATE_LABELS[name][int(logits.argmax(-1)[0])]
                                         for name,logits in state_logits.items()}
        if output.get('type_logits') is not None:
            from src.data.operations import OPERATION_CLASSES
            result['predicted_operation_probabilities'] = dict(zip(OPERATION_CLASSES,output['type_logits'][0].sigmoid().cpu().tolist()))
        if output['relevance_logits'] is not None:
            ids = item['diagnostics']['turn_ids']
            mask = output['relevance_mask'][0].tolist()
            result['turn_scores'] = {str(t):float(p) for t,p,ok in zip(ids,output['relevance_logits'].sigmoid()[0].tolist(),mask) if ok}
            result['masked_turn_ids'] = [str(t) for t,ok in zip(ids,mask) if not ok]
        if output.get('operation_logits') is not None:
            from src.data.operations import OPERATION_CLASSES
            result['predicted_turn_operations'] = dict(zip(map(str,item['diagnostics']['turn_ids']),
                [OPERATION_CLASSES[i] for i in output['operation_logits'][0].argmax(-1).tolist()]))
        embedding = None
        if model_name in ('A','C','C+'):
            embedding = {k:result[k] for k in ('sample_id','split','group_id','model','seed','checkpoint_hash')}
            embedding.update(cls=output['cls_representation'][0].cpu().tolist(),
                             representation='last_layer_cls',encoder_frozen=True)
        yield result, embedding
