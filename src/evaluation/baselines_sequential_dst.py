"""SOM-DST에서 착안한 선택적 다중 규칙 예측 메모리 baseline이다.

slot 값 생성의 재현 구현은 아니다. forward는 gold를 사용하지 않으며, operation과 state 주석은 training_loss에서만 받는다."""
import torch
from torch import nn
from torch.nn import functional as F
from src.models.predictor import ModernBertPredictor
from src.models.dataset import STATE_LABELS, IGNORE_INDEX
from src.data.operations import OPERATION_CLASSES


class SequentialDST(nn.Module):
    """turn별 예측 연산으로 다중 규칙 메모리를 갱신하고 요청별 위험을 계산한다."""
    variant = 'P2'
    def __init__(self, encoder, memory_slots=16):
        super().__init__()
        if memory_slots < 2:
            raise ValueError('P2 needs multiple rule slots')
        self.encoder, self.memory_slots = encoder, memory_slots
        h = encoder.config.hidden_size
        self.operation_head = nn.Linear(2*h, 6)
        self.address_head = nn.Linear(2*h, memory_slots)
        self.parent_head = nn.Linear(2*h, memory_slots)
        self.candidate = nn.Linear(2*h, h)
        self.scope_head = nn.Linear(h, h)
        self.priority_head = nn.Linear(h, 1)
        self.status_embedding = nn.Linear(3, h, bias=False)
        self.state_heads = nn.ModuleDict({name: nn.Linear(h, len(classes)) for name, classes in STATE_LABELS.items()})
        self.failure_head = nn.Linear(2*h, 1)

    def forward(self, input_ids, attention_mask, *, turn_spans, turn_mask, request_span, request_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        turns, visible = ModernBertPredictor.pool_turns(hidden, turn_spans, attention_mask)
        request, complete = ModernBertPredictor.pool_turns(hidden, request_span.unsqueeze(1), attention_mask)
        if not (complete.squeeze(1) & request_mask).all():
            raise ValueError('P2 requires an intact final request')
        b, t, h = turns.shape
        memory = hidden.new_zeros(b, self.memory_slots, h)
        scope = torch.zeros_like(memory)
        status = hidden.new_zeros(b, self.memory_slots, 3)  # 메모리 상태 index는 미등록, 활성, 철회 순서다.
        priority = hidden.new_zeros(b, self.memory_slots, 1)
        effective_time = torch.zeros_like(priority)
        lineage = hidden.new_zeros(b, self.memory_slots, self.memory_slots)
        operations, states = [], {name: [] for name in STATE_LABELS}
        for i in range(t):
            previous = memory + scope + lineage.bmm(memory) + self.status_embedding(status)
            context = torch.cat((turns[:, i], previous.mean(1)), -1)
            logits = self.operation_head(context)
            probabilities = logits.softmax(-1)
            # straight-through 예측으로 학습과 추론에서 같은 hard rollout을 쓴다. gold 메모리나 teacher forcing은 사용하지 않는다.
            hard = F.one_hot(probabilities.argmax(-1), 6).to(hidden.dtype)
            op = hard + probabilities - probabilities.detach() if self.training else hard
            address_p = self.address_head(context).softmax(-1)
            address_hard = F.one_hot(address_p.argmax(-1), self.memory_slots).to(hidden.dtype)
            address = address_hard + address_p - address_p.detach() if self.training else address_hard
            valid = (visible[:, i] & turn_mask[:, i]).to(hidden.dtype)
            # REVOKE는 내용 대신 상태만 바꾸고 NO_CHANGE는 모든 필드를 유지한다. 나머지 연산은 내용을 갱신한다.
            write = address * (op[:, 0] + op[:, 1] + op[:, 3] + op[:, 4]).unsqueeze(-1) * valid.unsqueeze(-1)
            revoke = address * op[:, 2:3] * valid.unsqueeze(-1)
            candidate = self.candidate(context).tanh()
            memory = memory*(1-write.unsqueeze(-1)) + candidate.unsqueeze(1)*write.unsqueeze(-1)
            scope = scope*(1-write.unsqueeze(-1)) + self.scope_head(candidate).unsqueeze(1)*write.unsqueeze(-1)
            priority = priority*(1-write.unsqueeze(-1)) + self.priority_head(candidate).unsqueeze(1)*write.unsqueeze(-1)
            effective_time = effective_time*(1-write.unsqueeze(-1)) + (i+1)*write.unsqueeze(-1)
            status = status*(1-(write+revoke).unsqueeze(-1)) + write.unsqueeze(-1)*hidden.new_tensor([0,1,0]) + revoke.unsqueeze(-1)*hidden.new_tensor([0,0,1])
            parent = self.parent_head(context).softmax(-1)
            linked = address * (op[:, 1]+op[:, 3]+op[:, 4]).unsqueeze(-1)*valid.unsqueeze(-1)
            lineage = lineage*(1-linked.unsqueeze(-1)) + linked.unsqueeze(-1)*parent.unsqueeze(1)
            representation = (memory + lineage.bmm(memory) + self.status_embedding(status)).mean(1)
            operations.append(logits)
            for name, head in self.state_heads.items():
                states[name].append(head(representation))
        # 범위·priority·시간·규칙 상태를 별도 메모리 축으로 두고, 요청별 조회에도 활성·철회 상태를 반영한다.
        weights = ((scope*request).sum(-1)/(h**.5) + priority.squeeze(-1) + effective_time.squeeze(-1)/max(t,1) + status[:,:,1]-status[:,:,2]).softmax(-1)
        final = ((memory + lineage.bmm(memory) + self.status_embedding(status))*weights.unsqueeze(-1)).sum(1)
        return dict(failure_logits=self.failure_head(torch.cat((final, request[:,0]),-1)).squeeze(-1),
            final_state_logits={name:head(final) for name,head in self.state_heads.items()},
            relevance_logits=None,
            operation_logits=torch.stack(operations,1) if operations else hidden.new_empty(b,0,6),
            state_logits={name: torch.stack(values,1) if values else hidden.new_empty(b,0,len(STATE_LABELS[name])) for name,values in states.items()},
            memory=dict(rules=memory,scope=scope,status=status,priority=priority,effective_time=effective_time,lineage=lineage))


def offline_labels(record, turn_ids):
    """각 접두 trace를 A oracle로 평가한다. 텍스트에서 label을 추측하지 않는다."""
    from src.rules.engine import evaluate_policy
    from src.rules.schema import ENGINE_VERSION
    trace = record['operation_trace']
    operation_labels, states = [], {k: [] for k in STATE_LABELS}
    for turn in turn_ids:
        events = [e for e in trace if e['effective_turn'] <= turn]
        current = [e['op'] for e in events if e['effective_turn'] == turn and e['op'] != 'NO_CHANGE']
        initial = [r for r in record['initial_rules'] if r.get('effective_turn', 0) <= turn]
        if any(r.get('effective_turn',0) == turn for r in initial):
            current.append('ADD')
        # 같은 turn에 서로 다른 operation 유형이 있으면 단일 type label을 만들지 않고 무시한다.
        operation_labels.append(OPERATION_CLASSES.index(current[0]) if len(set(current)) == 1 else (5 if not current else IGNORE_INDEX))
        from src.data.oracle_adapter import canonical_request
        request = canonical_request(record['request'])
        result = evaluate_policy(initial, events, request, turn+1, ENGINE_VERSION)
        for field, classes in STATE_LABELS.items():
            value = (result.get('gold_state') or {}).get(field)
            ambiguous = field in (result.get('gold_state') or {}).get('ambiguous_fields', [])
            states[field].append(classes.index(value) if value in classes and not ambiguous else IGNORE_INDEX)
    return operation_labels, states


def training_loss(output, failure_labels, operation_labels, state_labels, weights):
    zero = output['failure_logits'].sum()*0
    def ce(logits, labels):
        mask = labels != IGNORE_INDEX
        return F.cross_entropy(logits[mask], labels[mask]) if mask.any() else zero
    operation = ce(output['operation_logits'], operation_labels)
    observed = [ce(output['state_logits'][k], v) for k,v in state_labels.items() if (v != IGNORE_INDEX).any()]
    state = torch.stack(observed).mean() if observed else zero
    mask = failure_labels != IGNORE_INDEX
    failure = F.binary_cross_entropy_with_logits(output['failure_logits'][mask], failure_labels[mask]) if mask.any() else zero
    return weights['operation']*operation + weights['state']*state + weights['failure']*failure
