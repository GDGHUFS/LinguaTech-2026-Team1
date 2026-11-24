"""저장 점수 adapter 검증에 C의 소형 tokenizer와 encoder fixture를 재사용한다."""
import re
from types import SimpleNamespace
import pytest

torch=pytest.importorskip('torch')
from src.models.predictor import ModernBertPredictor
from src.evaluation.export import predict_records


class WordTokenizer:
    # hge-1113-b-tests의 tests/test_dataset_predictor.py에서 가져온 fixture다.
    is_fast=True
    padding_side=truncation_side='right'
    def __call__(self,text,return_offsets_mapping=False,truncation=False,max_length=None):
        offsets=[(0,0)]+[m.span() for m in re.finditer(r'\S+',text)]
        if truncation and max_length is not None:
            offsets=offsets[:max_length]
        encoded={'input_ids':[1]+[2+(hash(text[a:b])%50) for a,b in offsets[1:]],'attention_mask':[1]*len(offsets)}
        if return_offsets_mapping:
            encoded['offset_mapping']=offsets
        return encoded
    def pad(self,items,return_tensors='pt'):
        width=max(len(i['input_ids']) for i in items)
        return {key:torch.tensor([i[key]+[0]*(width-len(i[key])) for i in items]) for key in ('input_ids','attention_mask')}


class TinyEncoder(torch.nn.Module):
    # C fixture를 재사용하며 사전 학습 가중치나 네트워크는 사용하지 않는다.
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(hidden_size=8,to_dict=lambda:{'hidden_size':8})
        self.embedding=torch.nn.Embedding(64,8)
    def forward(self,input_ids,attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_export_never_forwards_gold_and_freezes_encoder():
    class Spy(ModernBertPredictor):
        def forward(self,**kwargs):
            assert not set(kwargs)&{'failure_labels','state_labels','relevance_labels','gold_state','target_decision'}
            return super().forward(**kwargs)
    model=Spy('C+',encoder=TinyEncoder())
    row=dict(sample_id='fixture',split='id_test',group_id='fixture-group',request_turn_id=1,
             turns=[dict(turn_id=0,role='user',text='I revoke permission'),dict(turn_id=1,role='user',text='Send the email')],
             final_request='Send the email',gold_state={'authorization':'DENY'},target_decision='ALLOW')
    score,embedding=list(predict_records(model,WordTokenizer(),[row],model_name='C+',seed=42,checkpoint_hash='fixture'))[0]
    assert score['sample_id']=='fixture' and 0<=score['failure_probability']<=1
    assert score['predicted_state'] and set(score['turn_scores'])=={'0'}
    assert len(embedding['cls'])==8 and embedding['encoder_frozen']
    assert all(not p.requires_grad for p in model.parameters())
