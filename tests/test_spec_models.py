"""CPU의 합성 입력으로 모델 구조와 생성 전 표현 추출을 검사한다."""
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from test_evaluation_export import TinyEncoder,WordTokenizer
from src.models.predictor import ModernBertPredictor
from src.models.dataset import DialogueDataset,DialogueCollator
from src.data.labeling import LocalTarget


def batch():
    record=dict(turns=[dict(turn_id=0,role='user',text='I revoke the permission.')],
        final_request='May the email be sent?',operation_trace=[dict(op='REVOKE')],failure_label_unsafe=1)
    tokenizer=WordTokenizer()
    item=DialogueDataset([record],tokenizer,128)[0]
    return DialogueCollator(tokenizer)([item])


def test_b_pred_conditions_on_sigmoid_and_backpropagates_failure_to_type():
    model=ModernBertPredictor('B_pred',encoder=TinyEncoder(),dropout=0)
    b=batch()
    captured=[]
    handle=model.failure_head.register_forward_pre_hook(lambda _,args:captured.append(args[0]))
    output=model(**b)
    handle.remove()
    assert torch.allclose(captured[0][:,-6:],output['type_logits'].sigmoid())
    assert torch.allclose(output['loss'],output['losses']['failure']+.5*output['losses']['type'])
    output['losses']['failure'].backward()
    assert model.type_head.weight.grad.abs().sum()>0
    b['type_labels'].zero_()
    assert torch.equal(model(**b)['failure_logits'],output['failure_logits'])
    with pytest.raises(ValueError,match='diagnostic-only'):
        model(**b,oracle_type_features=torch.zeros(1,6))


def test_b_oracle_is_explicit_and_same_failure_feature_dimension():
    model=ModernBertPredictor('B_oracle',encoder=TinyEncoder())
    b=batch()
    with pytest.raises(ValueError,match='explicit'):
        model(**b)
    assert model(**b,oracle_type_features=b['type_labels'])['failure_logits'].shape==(1,)
    assert model.failure_head.in_features==ModernBertPredictor('B_pred',encoder=TinyEncoder()).failure_head.in_features
    with pytest.raises(ValueError,match='separate'):
        model.predict([], 'May I share?', WordTokenizer())


def test_cplus_auxiliary_labels_are_losses_only():
    model=ModernBertPredictor('C+',encoder=TinyEncoder(),dropout=0)
    b=batch()
    b['state_labels']={k:torch.tensor([0]) for k in b['state_labels']}
    b['relevance_labels'].fill_(1)
    output=model(**b)
    assert output['losses']['state']>0 and output['losses']['relevance']>0
    assert torch.allclose(output['loss'],output['losses']['failure']+.5*output['losses']['state']+.2*output['losses']['relevance'])
    b['state_labels']={k:torch.tensor([-100]) for k in b['state_labels']}
    b['relevance_labels'].fill_(-100)
    masked=model(**b)
    assert torch.equal(output['failure_logits'],masked['failure_logits'])
    assert masked['losses']['state']==masked['losses']['relevance']==0


def test_target_pregen_never_generates_and_uses_final_nonpadding():
    class FakeTarget:
        config=SimpleNamespace(num_hidden_layers=4)
        def eval(self): return self
        def __call__(self,**kw):
            assert kw['output_hidden_states'] and kw['use_cache'] is False
            return SimpleNamespace(hidden_states=tuple(torch.arange(12).reshape(1,4,3).float()+100*l for l in range(5)))
        def generate(self,**kw): raise AssertionError('P1 must not generate')
    target=object.__new__(LocalTarget)
    target.model=FakeTarget()
    target.tokenizer=SimpleNamespace(padding_side='right')
    target.encode_prompt=lambda prompt:dict(input_ids=torch.tensor([[3,4,0,0]]),attention_mask=torch.tensor([[1,1,0,0]]))
    vector,metadata=target.pre_generation('same prompt')
    assert vector==[203.,204.,205.]
    assert metadata['generation_tokens']==0 and metadata['token_position']==1 and metadata['layer_index']==2
    target.encode_prompt=lambda prompt:dict(input_ids=torch.tensor([[0,0,3,4]]),attention_mask=torch.tensor([[0,0,1,1]]))
    assert target.pre_generation('same prompt')[0]==[209.,210.,211.]


def test_common_training_budget_all_core_variants(tmp_path):
    import yaml
    from train import train_model,seed_everything
    config=yaml.safe_load(Path('configs/train.yaml').read_text(encoding='utf-8'))
    config.update(max_epochs=1,per_device_batch_size=1,gradient_accumulation_steps=2,effective_batch_size=2)
    tokenizer=WordTokenizer()
    rows=[dict(turns=[dict(turn_id=0,role='user',text='Allow sharing')],final_request='Share email?',
               operation_trace=[dict(op='ADD')],failure_label_unsafe=i%2) for i in range(4)]
    items=[DialogueDataset(rows,tokenizer,128)[i] for i in range(4)]
    for mode,variant in [('A','A'),('B_pred','B_pred'),('C','C'),('C_plus','C+')]:
        seed_everything(42)
        model=ModernBertPredictor(variant,encoder=TinyEncoder())
        state=train_model(model,items,items,DialogueCollator(tokenizer),config,
            dict(mode=mode,seed=42,data_kind='synthetic_test_only'),tmp_path/mode)
        assert state['global_step']==2


def test_p2_no_gold_forward_and_nochange_carries_memory():
    from src.evaluation.baselines_sequential_dst import SequentialDST
    model=SequentialDST(TinyEncoder(),memory_slots=3).eval()
    with torch.no_grad():
        model.operation_head.weight.zero_()
        model.operation_head.bias.fill_(-100)
        model.operation_head.bias[5]=100
    b=batch()
    features={k:v for k,v in b.items() if k in ('input_ids','attention_mask','turn_spans','turn_mask','request_span','request_mask')}
    output=model(**features)
    assert all(torch.count_nonzero(value)==0 for value in output['memory'].values())
    with pytest.raises(TypeError):
        model(**features,gold_state={})
    model.train()
    output=model(**features)
    output['failure_logits'].sum().backward()
    assert model.encoder.embedding.weight.grad is not None


def test_p2_offline_intermediate_labels_come_from_a_oracle():
    from src.evaluation.baselines_sequential_dst import offline_labels
    from src.data.mutations import make_variants
    from src.data.oracle_adapter import annotate_dialogue
    seed=dict(status='eligible_pending_review',seed_id='fixture',source_seed_id='fixture',source_dataset='fixture',
        source_record_id='fixture',group_id='fixture',source_provenance={'fixture':True},policy_context='synthetic',
        data_type='email',subject='owner',sender='owner',recipient='organizer',purpose='event')
    sample=next(v for v in make_variants(seed) if v['capability_id']=='CAP05')
    dialogue=dict(turns=[dict(turn_id=i,role='user',text=t) for i,t in enumerate(['Allow email','Revoke it','Grant again','Share email?'])],
        initial_rule_turns={sample['initial_rules'][0]['rule_id']:0},operation_turns={'CAP05_op0':1,'CAP05_op1':2})
    record=annotate_dialogue(sample,dialogue)
    operations,states=offline_labels(record,[0,1,2])
    assert operations==[0,2,4]
    assert states['rule_status']==[0,1,0]
