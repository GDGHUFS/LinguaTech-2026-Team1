"""A의 apply/replay/decide 구조를 바탕으로 통합 정책 oracle을 구현한다.

구체성, override 계보, 최근 effective_turn 순으로 판단한다. 해소되지 않은 충돌은 INVALID 주석이며 DENY나 CLARIFY 학습 label로 만들지 않는다."""
from __future__ import annotations
from dataclasses import replace
from itertools import product
from .schema import (ANY, ENGINE_VERSION, Authorization, Decision, Gold, Request,
                     Rule, RuleStatus, State, Transition, TransitionType)

OTHER = "__other__"
ALLOW, DENY = Authorization.ALLOW, Authorization.DENY
T = TransitionType
MODIFIABLE = {"data_type", "recipient", "purpose", "subject", "sender", "scope",
              "authorization", "expiration", "expires_turn"}


class TransitionError(ValueError):
    pass


def _require(cond, msg):
    if not cond:
        raise TransitionError(msg)


def _get(state, rule_id):
    _require(rule_id in state, f"unknown target_id: {rule_id}")
    return state[rule_id]


def _narrower(child, parent):
    a, b = child.scope_values, parent.scope_values
    pairs = [(a.get(k, ANY), b.get(k, ANY)) for k in a.keys() | b.keys()]
    return all(p in (ANY, c) for c, p in pairs) and any(c != p for c, p in pairs)


def apply(state, tr):
    """연산 하나를 적용한 상태 복사본을 반환한다. 원본 상태는 바꾸지 않으며 이전 event와 변경 필드를 기록한다."""
    state = state.copy() if isinstance(state, State) else State(state)
    t, kind = tr.turn, tr.type
    _require(isinstance(kind, T), "Internal operations must be canonical")
    eid = tr.event_id or f"event:{t}:{kind.value}:{tr.rule.rule_id if tr.rule else tr.target_id or 'narrative'}"
    _require(eid not in state.events, f"duplicate event_id: {eid}")
    before, after, dependencies = None, None, []
    rid = tr.rule.rule_id if kind in (T.ADD, T.EXCEPTION) and tr.rule else tr.target_id
    if kind == T.NO_CHANGE:
        _require(tr.rule is None and not tr.changes and tr.target_id is None, "NO_CHANGE cannot mutate a rule")
    elif kind in (T.ADD, T.EXCEPTION):
        rule = tr.rule
        _require(rule is not None, f"turn {t}: {kind.value} needs a rule")
        _require(rule.rule_id not in state, f"duplicate rule_id: {rule.rule_id}")
        _require(rule.status == RuleStatus.ACTIVE, "New rule must be active")
        _require(not tr.changes, "ADD/EXCEPTION cannot carry changes")
        parent_id = None
        if kind == T.EXCEPTION:
            parent = _get(state, tr.target_id)
            _require(parent.status == RuleStatus.ACTIVE, "Exception parent is not active")
            _require(parent.effective_turn <= t, "Exception predates parent")
            _require(not parent.is_expired(t) if not isinstance(parent.expiration, str) else True,
                     "Exception parent is expired")
            _require(rule.authorization != parent.authorization, "Exception must flip parent authorization")
            _require(_narrower(rule, parent), "Exception must be narrower than parent")
            parent_id = parent.rule_id
            dependencies.append(state.latest[parent_id])
        else:
            _require(tr.target_id is None, "ADD cannot override another rule")
        after = replace(rule, parent_id=parent_id, effective_turn=t, updated_turn=t,
                        history=((t, kind),), lineage=tuple(dependencies))
    else:
        before = _get(state, tr.target_id)
        _require(before.effective_turn <= t, "Operation predates rule")
        dependencies.append(state.latest[before.rule_id])
        changes = dict(tr.changes)
        if kind == T.REVOKE:
            _require(tr.rule is None and not changes, "REVOKE only references a permission")
            _require(before.status == RuleStatus.ACTIVE and before.authorization == ALLOW,
                     "REVOKE needs an active ALLOW rule")
            after = replace(before, status=RuleStatus.REVOKED, updated_turn=t)
        elif kind == T.REINSTATE:
            _require(before.status == RuleStatus.REVOKED, "REINSTATE needs a revoked rule")
            _require(tr.rule is None, "Use changes to specify reapproval conditions")
            _require(not set(changes) - MODIFIABLE, "Unsupported reapproval field")
            _require(changes.get("authorization", "ALLOW") in ("ALLOW", ALLOW), "REINSTATE creates ALLOW")
            changes["authorization"] = ALLOW
            after = replace(before, **changes, status=RuleStatus.ACTIVE, effective_turn=t, updated_turn=t)
        elif kind == T.MODIFY:
            _require(before.status == RuleStatus.ACTIVE, "MODIFY needs an active rule")
            _require(tr.rule is None and changes and not set(changes) - MODIFIABLE, "Unsupported or empty MODIFY")
            if "authorization" in changes:
                changes["authorization"] = Authorization(changes["authorization"])
            _require(any(getattr(before, k) != v for k, v in changes.items()), "MODIFY changes nothing")
            after = replace(before, **changes, effective_turn=t, updated_turn=t)
        else:
            raise TransitionError(f"Unsupported operation: {kind}")
        after = replace(after, history=before.history + ((t, kind),),
                        lineage=before.lineage + (dependencies[0],))
    state.events[eid] = {"event_id": eid, "op": kind.value, "effective_turn": t,
                         "rule_id": rid, "predecessor_event_ids": dependencies,
                         "before": before.to_dict() if before else None,
                         "after": after.to_dict() if after else None,
                         "changed_fields": sorted(tr.changes)}
    state.edges.update((d, eid) for d in dependencies)
    if after is not None:
        state[after.rule_id] = after
        state.latest[after.rule_id] = eid
    return state


def replay(transitions, state=None):
    """turn 순서대로 연산을 재생한 상태 복사본을 반환한다. 역순 전이는 TransitionError로 거부한다."""
    state = state.copy() if isinstance(state, State) else State(state or {})
    last = max((e["effective_turn"] for e in state.events.values()), default=-1)
    for tr in transitions:
        _require(tr.turn >= last, f"transitions out of order at turn {tr.turn}")
        last = tr.turn
        state = apply(state, tr)
    return state


def _matches(rule, scope):
    return all(v == ANY or scope.get(k) == v for k, v in rule.scope_values.items())


def _ancestors(state, event_ids):
    """판단 근거 event에서 선행 의존성을 역추적해 증거 전체를 모은다."""
    closure, pending = set(event_ids), list(event_ids)
    while pending:
        eid = pending.pop()
        for parent in state.events[eid]["predecessor_event_ids"]:
            if parent not in closure:
                closure.add(parent)
                pending.append(parent)
    return closure


def _resolve(state, active, scope):
    cands = [r for r in active if _matches(r, scope)]
    if not cands:
        return DENY, [], False
    specific = max(r.specificity for r in cands)
    cands = [r for r in cands if r.specificity == specific]
    shadowed = set()
    for r in cands:
        parent = r.parent_id
        while parent is not None:
            shadowed.add(parent)
            parent = state[parent].parent_id
    cands = [r for r in cands if r.rule_id not in shadowed]
    recent = max(r.effective_turn for r in cands)
    winners = [r for r in cands if r.effective_turn == recent]
    conflict = len({r.authorization for r in winners}) > 1
    return (None if conflict else winners[0].authorization), winners, conflict


def _result(state, req):
    """누락된 요청 필드의 가능한 판단을 비교하고 요청별 gold와 근거를 구성한다."""
    _require(all(e["effective_turn"] < req.turn for e in state.events.values()),
             "All policy events must precede current request")
    fields = set(req.scope_values) | {k for r in state.values() for k in r.scope_values}
    active = [r for r in state.values() if r.is_active(req.turn, req.request_time)]
    options, names = [], sorted(fields)
    for name in names:
        value = req.scope_values.get(name)
        values = {r.scope_values.get(name, ANY) for r in active} - {ANY}
        other = OTHER
        while other in values:
            other += "_"
        # 미지정 필드는 규칙에 등장한 값과 나머지 범주의 대표값으로 나눠 판단 차이를 확인한다.
        options.append([value] if value is not None else sorted(values) + [other])
    scopes = [dict(zip(names, combo)) for combo in product(*options)]
    outcomes = [_resolve(state, active, scope) for scope in scopes]
    conflict = any(o[2] for o in outcomes)
    missing = len({o[0] for o in outcomes}) > 1 and not conflict
    decision = None if conflict else "CLARIFY" if missing else outcomes[0][0].value
    basis_by_id, scope_labels, status_labels = {}, set(), set()
    for scope, (auth_value, winners, _) in zip(scopes, outcomes):
        basis_by_id.update((r.rule_id, r) for r in winners)
        # 철회·만료·범위 변경으로 과거 규칙이 더는 적용되지 않는 근거도 증거에 포함한다.
        blockers = []
        for r in state.values():
            if r in winners or (r in active and _matches(r, scope)):
                continue
            snapshots = [event['after'] for event in state.events.values()
                         if event['rule_id'] == r.rule_id and event['after']]
            prior_matches = [Rule.from_dict(snapshot) for snapshot in snapshots
                             if _matches(Rule.from_dict(snapshot), scope)]
            if not prior_matches:
                continue
            rank = max((p.specificity, p.effective_turn) for p in prior_matches)
            winner_rank = max(((p.specificity, p.effective_turn) for p in winners), default=(-1, -1))
            if not winners or (r.authorization != auth_value and rank >= winner_rank):
                blockers.append(r)
        if blockers:
            basis_by_id.update((r.rule_id, r) for r in blockers)
        scope_labels.add('MATCH' if winners or any(_matches(r, scope) for r in blockers) else 'MISMATCH')
        outcome_basis = winners + blockers
        status_labels.update('ACTIVE' if r.is_active(req.turn, req.request_time) else
                             'REVOKED' if r.status == RuleStatus.REVOKED else 'EXPIRED'
                             for r in outcome_basis)
        if not outcome_basis:
            status_labels.add('NONE')
    basis = list(basis_by_id.values())
    roots = [state.latest[r.rule_id] for r in basis]
    closure = _ancestors(state, roots)
    status = next(iter(status_labels)) if len(status_labels) == 1 else None
    scope_match = "INCOMPLETE" if missing else next(iter(scope_labels)) if len(scope_labels) == 1 else None
    ambiguous_fields = ["rule_status"] if status is None else []
    if scope_match is None:
        ambiguous_fields.append('scope_match')
    auth = None if conflict else "UNRESOLVED" if missing else decision
    if conflict:
        ambiguous_fields = ["authorization", "rule_status", "scope_match"]
        status = scope_match = None
    ordered = sorted(closure, key=lambda x: (state.events[x]["effective_turn"], x))
    trace = {"events": ordered, "edges": [list(edge) for edge in sorted(state.edges) if set(edge) <= closure],
             "justification_event_ids": sorted(roots), "event_details": [state.events[e] for e in ordered]}
    return {"gold_decision": decision,
            "gold_state": {"authorization": auth, "rule_status": status, "scope_match": scope_match,
                           "ambiguous_fields": ambiguous_fields},
            "supporting_rule_ids": sorted({state.events[e]["rule_id"] for e in closure if state.events[e]["rule_id"]}),
            "supporting_turn_ids": sorted({state.events[e]["effective_turn"] for e in closure if state.events[e]["effective_turn"] != req.turn}),
            "evidence_trace": trace,
            "ambiguity_flags": {"unresolved_conflict": conflict, "missing_context": missing,
                                "ambiguous_state": bool(ambiguous_fields), "state_fields": ambiguous_fields},
            "annotation_status": "INVALID" if conflict else "VALID", "engine_version": ENGINE_VERSION}


def decide(state, req, default=DENY):
    _require(default == DENY, "semantics-v1 fixes default DENY")
    result = _result(state, req)
    roots = result["evidence_trace"]["justification_event_ids"]
    events = [state.events[e] for e in roots]
    last = max(events, key=lambda e: (e["effective_turn"], e["event_id"])) if events else None
    winners = [r for r in state.values() if state.latest.get(r.rule_id) in roots and r.is_active(req.turn, req.request_time)]
    winner = winners[0] if len(winners) == 1 else None
    return Gold(Decision(result["gold_decision"]) if result["gold_decision"] else None,
                Authorization(result["gold_state"]["authorization"]) if result["gold_state"]["authorization"] else None,
                winner.rule_id if winner else None, winner.scope_values if winner else None,
                T(last["op"]) if last else None, last["effective_turn"] if last else None,
                tuple(result["supporting_rule_ids"]), result["ambiguity_flags"]["ambiguous_state"],
                result["annotation_status"], result)


def evaluate_policy(initial_rules, operation_trace, request, request_turn_id, semantics_version=ENGINE_VERSION):
    """명세서 4절의 구조화된 규칙·연산·요청을 받아 gold와 근거 trace를 반환한다.

    자연어에서 정책을 추론하지 않는다. 유효하지 않은 전이는 TransitionError로 거부하며, 해소되지 않은 규칙 충돌은 annotation_status=INVALID로 반환한다."""
    _require(semantics_version == ENGINE_VERSION, f"Unsupported semantics version: {semantics_version}")
    initial = []
    for raw in initial_rules:
        r = raw if isinstance(raw, Rule) else Rule.from_dict(raw)
        _require(r.rule_id not in {tr.rule.rule_id for tr in initial}, f'duplicate rule_id: {r.rule_id}')
        _require(r.status == RuleStatus.ACTIVE and not r.history and not r.lineage and r.parent_id is None,
                 "Initial rules must be fresh; express prior changes as events")
        initial.append(Transition(r.effective_turn, T.ADD, r, event_id=f"initial:{r.rule_id}"))
    state = replay(sorted(initial, key=lambda tr: tr.turn))
    transitions = [tr if isinstance(tr, Transition) else Transition.from_dict(tr) for tr in operation_trace]
    for raw, tr in zip(operation_trace, transitions):
        if isinstance(raw, dict) and raw.get("rule_id") is not None and tr.type in (T.ADD, T.EXCEPTION):
            _require(tr.rule is not None and raw["rule_id"] == tr.rule.rule_id, "Event rule_id differs from rule")
    state = replay(transitions, state)
    if isinstance(request, Request):
        _require(request.turn == request_turn_id, "Request turn mismatch")
        req = request
    else:
        _require("turn" not in request or request["turn"] == request_turn_id, "Request turn mismatch")
        req = Request.from_dict({**request, "turn": request_turn_id})
    return _result(state, req)


def compute_gold(scenario, default=DENY):
    """initial_rules를 포함한 시나리오를 통합 oracle 계약으로 평가하는 호환 진입점이다."""
    _require(default == DENY, "semantics-v1 fixes default DENY")
    request = scenario["request"]
    return evaluate_policy(scenario.get("initial_rules", []),
                           scenario.get("operation_trace", scenario.get("transitions", [])),
                           request, scenario.get("request_turn_id", request.get("turn")),
                           scenario.get("semantics_version", ENGINE_VERSION))
