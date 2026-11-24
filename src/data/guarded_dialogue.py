"""Closed-language policy rendering and independently replayable admission checks.

Qwen may select a surface form from a finite, semantics-preserving grammar.
It never supplies structure, mappings, a request, an event, or a label.
"""
from copy import deepcopy
import hashlib
import json
import re

from src.utils.reproducibility import content_hash

ARCHITECTURE = 'guarded_v1'
GRAMMAR_VERSION = 'policy-language-v1'
MISSING_RECIPIENT = 'The recipient of this new request is unspecified and may differ from every person named earlier.'


def quote(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def schedule(sample):
    initial, operations = {}, {}
    index = 1  # immutable, verbatim source background occupies turn zero
    for rule in sample['initial_rules']:
        initial[rule['rule_id']] = index
        index += 2
    for event in sample['operation_trace']:
        operations[event['operation_id']] = index
        index += 2
    return dict(initial_rule_turns=initial, operation_turns=operations, request_turn_id=max(3,index))


def with_schedule(sample):
    result = deepcopy(sample)
    result['generation_architecture'] = ARCHITECTURE
    result['policy_turn_plan'] = schedule(result)
    return result


def source_background(sample):
    if sample.get('source_normalization'):
        from src.data.automatic import source_scenario
        narrative = source_scenario(sample)['story'].partition('Question:')[0].strip()
    else:
        narrative = str(sample.get('policy_context') or '').partition('Question:')[0].strip()
    if not narrative:
        raise ValueError('source_background_missing')
    return ('Historical source background, quoted without changes:\n'+narrative+
        '\nThe following instructions describe a separate, later hypothetical privacy policy. '
        'They do not assert that the source subject originally consented to sharing. '
        'Policy scopes are conditions, not claims that anyone has additional personal information.')


def information(rule):
    return ('information described as '+quote(rule['data_type'])+' about '+quote(rule['subject'])+
        ' with '+quote(rule['recipient'])+', solely for '+quote(rule['purpose']))


def conditions(rule):
    items = []
    for key,value in sorted((rule.get('scope') or {}).items()):
        items.append(key+' must equal '+quote(value))
    if rule.get('expiration') is not None:
        items.append('the rule is valid through '+('turn ' if isinstance(rule['expiration'],int) else 'timestamp ')+quote(rule['expiration'])+' (inclusive), and expires afterward')
    if rule.get('expires_turn') is not None:
        items.append('the rule is valid through turn '+quote(rule['expires_turn'])+' (inclusive), and expires afterward')
    return (' Additional conditions: '+'; '.join(items)+'.') if items else ''


def rule_forms(rule, alias):
    tail = information(rule)+'.'+conditions(rule)
    subject, sender = quote(rule['subject']), quote(rule['sender'])
    if rule['authorization'] == 'ALLOW':
        return [f'Under policy {alias}, {subject} explicitly permits {sender} to share '+tail,
                f'For policy {alias}, {subject} expressly authorizes {sender} to share '+tail,
                f'{subject} grants {sender} permission under policy {alias} to share '+tail]
    if rule['authorization'] == 'DENY':
        return [f'Under policy {alias}, {subject} explicitly prohibits {sender} from sharing '+tail,
                f'For policy {alias}, {subject} expressly forbids {sender} from sharing '+tail,
                f'{subject} sets a prohibition under policy {alias}: {sender} must not share '+tail]
    raise ValueError('unsupported_authorization')


def request_text(sample):
    request = sample['request']
    recipient = quote(request['recipient']) if request.get('recipient') is not None else 'an unspecified recipient'
    body = (quote(request['sender'])+' share information described as '+quote(request['data_type'])+
        ' about '+quote(request['subject'])+' from the source background with '+recipient+
        ', for '+quote(request['purpose']))
    extras = {k:v for k,v in request.items() if k not in ('sender','subject','data_type','recipient','purpose','scope','request_time','text')}
    extras.update(request.get('scope') or {})
    if extras:
        body += ', under these request conditions: '+', '.join(k+'='+quote(v) for k,v in sorted(extras.items()))
    if request.get('request_time') is not None:
        body += ', at '+quote(request['request_time'])
    style = sample.get('request_style')
    conditional = quote(request['sender'])+' were to share'+body[len(quote(request['sender'])+' share'):]
    indirect = quote(request['sender'])+' to share'+body[len(quote(request['sender'])+' share'):]
    text = ('What would happen if '+conditional+'?' if style == 'question_embedding' else
            'Would sharing be permitted if '+conditional+'?' if style == 'paraphrase' else
            'Please determine whether the policy permits '+indirect+'.' if style == 'indirect' else
            'May '+body+'?')
    return (MISSING_RECIPIENT+' '+text) if request.get('recipient') is None else text


def blueprint(sample):
    """Render from engine-validated before/after snapshots, not guessed event meanings."""
    from src.data.oracle_adapter import _inputs, evaluate_structured
    from src.rules.engine import replay, apply
    from src.rules.schema import Rule, Transition, TransitionType
    trace = with_schedule(sample)
    if sample.get('policy_turn_plan') != trace['policy_turn_plan']:
        raise ValueError('policy_turn_plan_changed')
    oracle = evaluate_structured(trace)
    if oracle['annotation_status'] != 'VALID':
        raise ValueError('invalid_structured_oracle')
    initial, events, _, _ = _inputs(trace)
    ids = [r['rule_id'] for r in initial]+[e['rule']['rule_id'] for e in events if e.get('rule')]
    aliases = {rid:chr(65+i) for i,rid in enumerate(dict.fromkeys(ids))}
    if len(aliases)>26:
        raise ValueError('too_many_policies_for_closed_grammar')
    turns, allowed, meanings = [], {}, {}
    def add(forms, meaning, role='user'):
        index = len(turns)
        allowed[index] = forms
        meanings[index] = deepcopy(meaning)
        # Lexical choice depends on source family, not on a Gold label.
        salt = sample.get('group_id','')+':'+str(index)+':42'
        choice = int(hashlib.sha256(salt.encode()).hexdigest(),16)%len(forms)
        turns.append(dict(turn_id=index,role=role,text=forms[choice]))
    add([source_background(sample)],dict(kind='source_background'))
    state = replay([])
    def acknowledge():
        add(['I have noted that statement.','That statement is recorded.'],dict(kind='acknowledgement'),role='assistant')
    for raw in initial:
        rid = raw['rule_id']
        add(rule_forms(raw,aliases[rid]),dict(kind='initial_rule',rule=raw))
        state = apply(state,Transition(raw['effective_turn'],TransitionType.ADD,
            Rule.from_dict(raw),event_id='initial:'+rid))
        acknowledge()
    for raw in events:
        tr = Transition.from_dict(raw)
        before = state.get(tr.target_id)
        updated = apply(state,tr)  # the existing engine enforces active ALLOW / revoked preconditions
        kind = raw['op']
        rid = raw.get('rule_id') or raw.get('target_id')
        alias = aliases.get(rid)
        if kind == 'ADD':
            forms = ['A new rule is now established. '+text for text in rule_forms(raw['rule'],alias)]
        elif kind == 'EXCEPTION':
            parent_alias = aliases[raw['target_id']]
            forms = [f'A narrower exception, policy {alias}, is established to general policy {parent_alias}. '+
                text+' It is an exception only within its stated narrower conditions.'
                for text in rule_forms(raw['rule'],alias)]
        elif kind == 'REVOKE':
            if not before.is_active(tr.turn,sample['request'].get('request_time')):
                raise ValueError('revoke_requires_effective_permission')
            tail = (f' the active permission under policy {alias} for '+quote(before.sender)+' to share '+
                information(before.to_dict())+'. That permission under this policy is withdrawn; other policies are unchanged.')
            forms = [quote(before.subject)+' now '+verb+tail for verb in ('revokes','withdraws','rescinds')]
        elif kind == 'REINSTATE':
            after = updated[rid].to_dict()
            forms = [f'After the revocation, a new active permission replaces the revoked authorization in policy {alias}. '+text
                for text in rule_forms(after,alias)]
        elif kind == 'MODIFY':
            after = updated[rid].to_dict()
            old = before.to_dict()
            def display(key,value):
                return {'ALLOW':'permission to share','DENY':'prohibition on sharing'}.get(value,value) if key=='authorization' else value
            replacements = '; '.join(k+' changes from '+quote(display(k,old.get(k)))+' to '+quote(display(k,v)) for k,v in sorted(raw['changes'].items()))
            forms = [f'Policy {alias} is amended: {replacements}. '+text+
                ' This is a revision of that same policy; all unmentioned conditions remain unchanged.'
                +(' The changed information category is a hypothetical policy scope, not an additional source fact.' if 'data_type' in raw['changes'] else '')
                for text in rule_forms(after,alias)]
        elif kind == 'NO_CHANGE':
            forms = ['I will pause here. This observation changes no policy.',
                     'I have no further contextual details to add. No sharing rule is added, changed, or withdrawn.']
        else:
            raise ValueError('unsupported_policy_operation')
        after = updated.get(rid)
        add(forms,dict(kind='operation',event=raw,
            before=before.to_dict() if before else None,after=after.to_dict() if after else None))
        state = updated
        acknowledge()
    while len(turns)<trace['policy_turn_plan']['request_turn_id']:
        add(['The stated policy has been recorded.'],dict(kind='acknowledgement'),role='assistant')
    final = request_text(sample)
    add([final],dict(kind='request',request=deepcopy(sample['request'])))
    dialogue = dict(turns=turns, final_request=final,
        initial_rule_turns=deepcopy(trace['policy_turn_plan']['initial_rule_turns']),
        operation_turns=deepcopy(trace['policy_turn_plan']['operation_turns']))
    program = dict(grammar_version=GRAMMAR_VERSION,trace_hash=content_hash(trace),
        allowed=allowed,meanings=meanings,dialogue=dialogue)
    program['plan_hash'] = content_hash(program)
    return program


def sentence_witness(text, meaning, aliases=None):
    """Separate lexical decoder for this closed language, not an LLM confidence flag."""
    if meaning['kind'] not in ('operation','initial_rule'):
        return {}, []
    errors, witness = [], {}
    if meaning['kind']=='operation':
        decoded = ('ADD' if text.startswith('A new rule is now established. ') else
            'EXCEPTION' if text.startswith('A narrower exception, policy ') else
            'REINSTATE' if text.startswith('After the revocation, a new active permission replaces ') else
            'MODIFY' if re.match(r'^Policy [A-Z] is amended: ',text) else
            'REVOKE' if re.match(r'^.+ now (revokes|withdraws|rescinds) the active permission under policy [A-Z] ',text) else
            'NO_CHANGE' if ('This observation changes no policy.' in text or
                'No sharing rule is added, changed, or withdrawn.' in text) else None)
        witness['decoded_operation'] = decoded
        if decoded!=meaning['event']['op']:
            errors.append('operation_not_expressed:'+meaning['event']['event_id'])
        if decoded=='NO_CHANGE':
            return witness,errors
        rule = meaning['before'] if decoded=='REVOKE' else meaning['after']
    else:
        rule = meaning['rule']
        decoded = None
    if not rule:
        return witness,errors+['missing_decoded_rule_scope']
    scope = {key:quote(rule[key]) for key in ('subject','sender','recipient','purpose','data_type')}
    if any(value not in text for value in scope.values()):
        errors.append('policy_scope_not_expressed')
    witness['scope_excerpts'] = scope
    string = r'"(?:\\.|[^"\\])*"'
    def part(name):
        return '(?P<'+name+'>'+string+')'
    tail = ('information described as '+part('data_type')+' about '+part('about')+
        ' with '+part('recipient')+', solely for '+part('purpose')+r'\.')
    if decoded=='REVOKE':
        patterns = [part('subject')+r' now (?:revokes|withdraws|rescinds) the active permission under policy (?P<alias>[A-Z]) for '+part('sender')+' to share '+tail]
    else:
        patterns = [
            r'Under policy (?P<alias>[A-Z]), '+part('subject')+r' explicitly (?:permits|prohibits) '+part('sender')+r' (?:to share|from sharing) '+tail,
            r'For policy (?P<alias>[A-Z]), '+part('subject')+r' expressly (?:authorizes|forbids) '+part('sender')+r' (?:to share|from sharing) '+tail,
            part('subject')+' grants '+part('sender')+r' permission under policy (?P<alias>[A-Z]) to share '+tail,
            part('subject')+r' sets a prohibition under policy (?P<alias>[A-Z]): '+part('sender')+' must not share '+tail]
    match = next((m for pattern in patterns if (m:=re.search(pattern,text))),None)
    if not match:
        errors.append('policy_roles_not_decodable')
    else:
        parsed = {key:json.loads(match[key]) for key in ('subject','sender','recipient','purpose','data_type','about')}
        if any(parsed[key]!=rule[key] for key in ('subject','sender','recipient','purpose','data_type')) or parsed['about']!=rule['subject']:
            errors.append('decoded_policy_roles_or_scope_mismatch')
        if aliases and match['alias']!=aliases[rule['rule_id']]:
            errors.append('decoded_policy_reference_mismatch')
        witness['decoded_scope'] = parsed
        witness['policy_reference'] = match['alias']
    if aliases and meaning['kind']=='operation':
        event = meaning['event']
        if event['op']=='EXCEPTION':
            header = re.match(r'A narrower exception, policy ([A-Z]), is established to general policy ([A-Z])\.',text)
            if not header or header[1]!=aliases[event['rule_id']] or header[2]!=aliases[event['target_id']]:
                errors.append('exception_parent_not_expressed')
        if event['op']=='MODIFY':
            if not text.startswith('Policy '+aliases[event['rule_id']]+' is amended: '):
                errors.append('modified_policy_reference_mismatch')
        if event['op']=='REINSTATE':
            if not text.startswith('After the revocation, a new active permission replaces the revoked authorization in policy '+aliases[event['rule_id']]+'.'):
                errors.append('reinstated_policy_reference_mismatch')
    if decoded!='REVOKE':
        extra_scope = dict(rule.get('scope') or {})
        for key in ('expiration','expires_turn'):
            if rule.get(key) is not None:
                extra_scope[key] = rule[key]
        if any(quote(value) not in text for value in extra_scope.values()):
            errors.append('policy_condition_not_expressed')
        if extra_scope:
            witness['condition_excerpts'] = {key:quote(value) for key,value in extra_scope.items()}
        allow = (' explicitly permits ',' expressly authorizes ',' permission under policy ')
        deny = (' explicitly prohibits ',' expressly forbids ',' must not share ')
        matched = [phrase for phrase in (allow if rule['authorization']=='ALLOW' else deny) if phrase in text]
        if not matched:
            errors.append('authorization_not_expressed')
        witness['authorization_excerpt'] = matched[0] if matched else None
    return witness,errors


def verify(sample, dialogue):
    """Evidence comes from actual sentence membership; unknown language stays UNVERIFIED."""
    from src.data.validators import validate_dialogue
    errors, evidence = [], []
    try:
        if sample.get('source_normalization'):
            from src.data.automatic import validate_scenario
            validate_scenario(sample)
        facts = sample.get('source_facts') or {}
        if facts:
            for rule in sample['initial_rules']+[o['rule'] for o in sample['operation_trace'] if 'rule' in o]:
                if any(rule.get(k)!=facts.get(k) for k in ('subject','sender','recipient')):
                    errors.append('unsupported_initial_source_scope')
                if rule.get('data_type')!=facts.get('data_type') and dict(rule_id=rule['rule_id'],
                        field='data_type',value=rule['data_type'],origin='synthetic_policy',asserted_source_fact=False) not in sample.get('hypothetical_initial_scopes',[]):
                    errors.append('unmarked_hypothetical_information_scope')
            known = {facts.get(k) for k in ('subject','sender','recipient')}
            for op in sample['operation_trace']:
                if 'recipient' in op.get('changes',{}) and op['changes']['recipient'] not in known:
                    errors.append('unsupported_synthetic_recipient')
            if sample['request'].get('recipient') is not None and sample['request']['recipient'] not in known:
                errors.append('unsupported_request_recipient')
        plan = blueprint(sample)
        ids = [r['rule_id'] for r in sample['initial_rules']]+[o['rule']['rule_id'] for o in sample['operation_trace'] if o.get('rule')]
        aliases = {rid:chr(65+i) for i,rid in enumerate(dict.fromkeys(ids))}
        errors += validate_dialogue(dialogue,sample)
        expected = plan['dialogue']
        for key in ('initial_rule_turns','operation_turns','final_request'):
            if dialogue.get(key) != expected[key]:
                errors.append('immutable_dialogue_field_changed:'+key)
        if len(dialogue.get('turns',[])) != len(expected['turns']):
            errors.append('immutable_turn_count_changed')
        for actual, fixed in zip(dialogue.get('turns',[]),expected['turns']):
            i = fixed['turn_id']
            if actual.get('turn_id')!=i or actual.get('role')!=fixed['role']:
                errors.append('immutable_turn_identity_changed:'+str(i))
            if actual.get('text') not in plan['allowed'][i]:
                errors.append('unverified_policy_sentence:'+str(i))
            else:
                witness,failures = sentence_witness(actual['text'],plan['meanings'][i],aliases)
                errors += failures
                evidence.append(dict(turn_id=i,quote=actual['text'],meaning=plan['meanings'][i],witness=witness))
    except (ValueError,TypeError,KeyError,AttributeError) as exc:
        errors.append('closed_grammar:'+str(exc))
        plan = None
    return dict(status='UNVERIFIED' if errors else 'VERIFIED',method='closed_grammar_membership',
        grammar_version=GRAMMAR_VERSION, errors=errors,evidence=evidence,
        plan_hash=plan['plan_hash'] if plan else None)


def paraphrase_request(plan, scope='single_policy_turn'):
    if scope == 'all_policy_turns':
        indices = [i for i,m in plan['meanings'].items()
            if m['kind'] in ('initial_rule','operation') and len(plan['allowed'][i])>1]
        if not indices:
            return None
        properties = {str(i):dict(type='string',enum=[text for text in plan['allowed'][i]
            if text != plan['dialogue']['turns'][i]['text']]) for i in indices}
        schema = dict(type='object',properties=dict(turns=dict(type='object',
            properties=properties,required=list(properties),additionalProperties=False)),
            required=['turns'],additionalProperties=False)
        prompt = ('Select a different approved wording for EVERY listed policy turn. '
            'Do not make a policy judgment or change facts, permissions, conditions or chronology. '
            'Return JSON with exactly one key turns, mapping each listed turn ID to one of its '
            'approved alternatives. Do not omit or add turns.\n'+json.dumps([
                dict(turn_id=str(i),original=plan['dialogue']['turns'][i]['text'],
                     alternatives=properties[str(i)]['enum']) for i in indices],ensure_ascii=False))
        return indices,prompt,schema
    if scope != 'single_policy_turn':
        raise ValueError('unsupported_paraphrase_scope')
    # One policy utterance per model call. No ID, mapping, label or final request is requested.
    candidates = [i for i,m in plan['meanings'].items() if m['kind']=='operation' and len(plan['allowed'][i])>1]
    if not candidates:
        candidates = [i for i,m in plan['meanings'].items() if m['kind']=='initial_rule']
    if not candidates:
        return None
    index = candidates[-1]
    original = plan['dialogue']['turns'][index]['text']
    alternatives = [t for t in plan['allowed'][index] if t != original]
    schema = dict(type='object',properties={'text':dict(type='string',enum=alternatives)},required=['text'],additionalProperties=False)
    prompt = ('Select one approved paraphrase of the following policy utterance. Do not make a policy judgment. '
        'Return only JSON with one key text, containing exactly one approved alternative.\n'
        'Original utterance: '+original+'\nApproved alternatives: '+json.dumps(alternatives,ensure_ascii=False))
    return index,prompt,schema


def paraphrase_candidate(plan, selected, parsed):
    """Apply only the exact turn set and finite alternatives requested from the model."""
    index,_,schema = selected
    candidate = deepcopy(plan['dialogue'])
    if isinstance(index,list):
        properties = schema['properties']['turns']['properties']
        if (not isinstance(parsed,dict) or set(parsed)!={'turns'}
                or not isinstance(parsed['turns'],dict) or set(parsed['turns'])!=set(properties)):
            raise ValueError('qwen_output_outside_closed_grammar')
        for key,field in properties.items():
            if not isinstance(parsed['turns'][key],str) or parsed['turns'][key] not in field['enum']:
                raise ValueError('qwen_output_outside_closed_grammar')
            candidate['turns'][int(key)]['text'] = parsed['turns'][key]
    else:
        if (not isinstance(parsed,dict) or set(parsed)!={'text'}
                or not isinstance(parsed['text'],str) or parsed['text'] not in schema['properties']['text']['enum']):
            raise ValueError('qwen_output_outside_closed_grammar')
        candidate['turns'][index]['text'] = parsed['text']
    return candidate


def paraphrase_coverage(plan, dialogue, scope):
    eligible = [i for i,m in plan['meanings'].items()
        if m['kind'] in ('initial_rule','operation') and len(plan['allowed'][i])>1]
    changed = [i for i in eligible
        if dialogue['turns'][i]['text'] != plan['dialogue']['turns'][i]['text']]
    return dict(scope=scope,eligible_turn_ids=eligible,changed_turn_ids=changed)


def execute(sample, client, attempts=1):
    from src.data.validators import validate_structure
    from src.data.oracle_adapter import evaluate_structured, annotate_dialogue
    from src.data.batch import model_json, diagnostic
    if type(attempts) is not int or not 0<=attempts<=3:
        raise ValueError('paraphrase_attempts_must_be_between_zero_and_three')
    trace = with_schedule(sample)
    errors = validate_structure(trace)
    record = dict(trace, structured_trace=deepcopy(trace), errors=errors,
        generation_method='template',fallback_used=False,failure_reason=[],qwen_attempts=[],
        qwen_validation_status='NOT_ATTEMPTED',generation_status='invalid_structure',
        validation_status='UNVERIFIED',policy_validation_status='UNVERIFIED',
        gold_training_eligible=False,failure_prediction_training_eligible=False)
    if errors:
        record['failure_reason'] = list(errors)
        return record
    oracle = evaluate_structured(trace)
    record['structured_oracle'] = oracle
    plan = blueprint(trace)
    output = deepcopy(plan['dialogue'])
    baseline_proof = verify(trace,output)
    if baseline_proof['status']!='VERIFIED':
        record.update(errors=baseline_proof['errors'],failure_reason=baseline_proof['errors'],
            policy_validation=baseline_proof)
        return record
    scope = trace.get('paraphrase_scope','single_policy_turn')
    selected = paraphrase_request(plan,scope)
    if client is not None and selected and attempts:
        index,prompt,schema = selected
        for _ in range(attempts):
            attempt = dict(prompt=prompt,prompt_hash=content_hash(prompt),schema=schema,
                schema_hash=content_hash(schema),validation_status='UNVERIFIED',raw=None)
            try:
                raw = client.complete_json(prompt,schema)
                attempt['raw'] = raw
                attempt['serving_provenance'] = deepcopy(client.serving_provenance)
                if getattr(client,'stage_callback',None):
                    client.stage_callback(trace,'qwen_paraphrase',raw)
                parsed = model_json(raw)
                candidate = paraphrase_candidate(plan,selected,parsed)
                proof = verify(trace,candidate)
                if proof['status']!='VERIFIED':
                    raise ValueError('qwen_meaning_unverified')
                output = candidate
                attempt.update(validation_status='VERIFIED')
                attempt['turn_ids' if isinstance(index,list) else 'turn_id'] = index
                record['generation_method'] = 'qwen_paraphrase'
                record['qwen_validation_status'] = 'VERIFIED'
            except Exception as exc:
                attempt['failure_reason'] = type(exc).__name__+':'+str(exc)[:240]
                record['failure_reason'].append(attempt['failure_reason'])
                record['qwen_validation_status'] = 'UNVERIFIED'
            if getattr(client,'serving_provenance',{}).get('status') != 'NOT VERIFIED':
                attempt['serving_provenance'] = deepcopy(client.serving_provenance)
            record['qwen_attempts'].append(attempt)
            if attempt['validation_status']=='VERIFIED':
                break
        record['fallback_used'] = record['generation_method']=='template'
    proof = verify(trace,output)
    if proof['status']!='VERIFIED':
        record.update(errors=proof['errors'],failure_reason=proof['errors'],policy_validation=proof)
        return record
    annotated = annotate_dialogue(trace,output)
    if any(annotated[k]!=oracle[k] for k in ('gold_decision','gold_state')):
        raise ValueError('dialogue_gold_changed')
    for key in ('initial_rules','operation_trace','gold_decision','gold_state','supporting_rule_ids',
        'supporting_turn_ids','evidence_trace','ambiguity_flags','annotation_status','engine_version',
        'versions','relevance_mask','oracle_verified'):
        record[key] = annotated[key]
    record.update(turns=output['turns'],dialogue=output,final_request=output['final_request'],
        request_turn_id=output['turns'][-1]['turn_id'],generation_status='generated',
        validation_status='semantic_pass_oracle_verified',policy_validation_status='VERIFIED',
        policy_validation=proof,dialogue_plan_hash=plan['plan_hash'],
        oracle_relation_status='passed',controlled_family_status='semantic_complete_oracle_verified',
        gold_training_eligible=True,errors=[],
        execution=dict(architecture=ARCHITECTURE,grammar_version=GRAMMAR_VERSION,
            seed=42,free_generation=False,model_id=client.model if client and record['qwen_attempts'] else None),
        diagnostics=[diagnostic(trace,'closed_grammar_validation','passed')])
    # Preserve the complete deterministic evidence instead of inventing model confidence.
    record['semantic_validation'] = deepcopy(proof)
    if 'paraphrase_scope' in trace:
        record['paraphrase_coverage'] = paraphrase_coverage(plan,output,scope)
    return record


def validate_record(record):
    """Replay stored proof, actual model text, immutable plan and engine annotations."""
    from src.data.automatic import validate_scenario
    from src.data.batch import model_json
    from src.data.oracle_adapter import annotate_dialogue
    from src.data.serving import validate_serving
    trace, dialogue = record['structured_trace'],record['dialogue']
    validate_scenario(trace)
    validate_scenario(record)
    immutable = ('sample_id','source_dataset','source_component','source_record_id','source_seed_id',
        'seed_id','group_id','variant_group_id','variant_id','parent_variant_id','capability_id',
        'template_family_id','request_style','purpose_provenance','source_facts','policy_temporal_boundary',
        'scope_intervention_provenance','hypothetical_initial_scopes','holdout_plan','holdout_plan_hash')
    if any(record.get(key)!=trace.get(key) for key in immutable):
        raise ValueError('immutable_symbolic_metadata_changed')
    if record.get('paraphrase_scope') != trace.get('paraphrase_scope'):
        raise ValueError('paraphrase_scope_changed')
    if trace.get('holdout_plan') is not None and trace.get('holdout_plan_hash')!=content_hash(trace['holdout_plan']):
        raise ValueError('holdout_plan_hash_changed')
    if trace.get('generation_architecture')!=ARCHITECTURE or record.get('generation_architecture')!=ARCHITECTURE:
        raise ValueError('guarded_architecture_changed')
    if (record.get('validation_status')!='semantic_pass_oracle_verified' or
            record.get('policy_validation_status')!='VERIFIED' or not record.get('gold_training_eligible')):
        raise ValueError('unverified_guarded_record')
    plan = blueprint(trace)
    proof = verify(trace,dialogue)
    if proof['status']!='VERIFIED' or proof!=record.get('policy_validation') or proof!=record.get('semantic_validation'):
        raise ValueError('guarded_semantic_evidence_changed')
    if record.get('dialogue_plan_hash')!=plan['plan_hash'] or record.get('policy_turn_plan')!=trace['policy_turn_plan']:
        raise ValueError('immutable_plan_changed')
    if record.get('turns')!=dialogue['turns'] or record.get('final_request')!=dialogue['final_request']:
        raise ValueError('stored_dialogue_changed')
    attempts = record.get('qwen_attempts',[])
    execution = record.get('execution') or {}
    if (execution.get('architecture')!=ARCHITECTURE or execution.get('grammar_version')!=GRAMMAR_VERSION
            or execution.get('seed')!=42 or execution.get('free_generation') is not False):
        raise ValueError('generation_execution_contract_changed')
    if len(attempts)>3:
        raise ValueError('unbounded_paraphrase_retries')
    scope = trace.get('paraphrase_scope','single_policy_turn')
    selected = paraphrase_request(plan,scope)
    if 'paraphrase_scope' in trace and record.get('paraphrase_coverage')!=paraphrase_coverage(plan,dialogue,scope):
        raise ValueError('paraphrase_coverage_changed')
    accepted = []
    for attempt in attempts:
        index,prompt,schema = selected
        if (attempt.get('prompt')!=prompt or attempt.get('prompt_hash')!=content_hash(prompt)
                or attempt.get('schema')!=schema or attempt.get('schema_hash')!=content_hash(schema)):
            raise ValueError('paraphrase_input_changed')
        if attempt.get('validation_status')=='VERIFIED':
            validate_serving(attempt.get('serving_provenance',{}))
            if execution.get('model_id')!=attempt['serving_provenance']['model_id']:
                raise ValueError('paraphrase_model_identity_changed')
            parsed = model_json(attempt['raw'])
            if attempt.get('turn_ids' if isinstance(index,list) else 'turn_id') != index:
                raise ValueError('paraphrase_turn_mapping_changed')
            candidate = paraphrase_candidate(plan,selected,parsed)
            accepted.append(candidate)
    if record.get('generation_method')=='qwen_paraphrase':
        if not accepted or dialogue!=accepted[-1] or record.get('fallback_used') or record.get('qwen_validation_status')!='VERIFIED':
            raise ValueError('qwen_paraphrase_provenance_mismatch')
    elif record.get('generation_method')=='template':
        if dialogue!=plan['dialogue'] or accepted:
            raise ValueError('template_changed_or_model_success_misreported')
        if bool(attempts)!=record.get('fallback_used'):
            raise ValueError('fallback_provenance_mismatch')
        if attempts and (not record.get('failure_reason') or record.get('qwen_validation_status')!='UNVERIFIED'):
            raise ValueError('fallback_missing_actual_failure')
    else:
        raise ValueError('unknown_generation_method')
    oracle = annotate_dialogue(trace,dialogue)
    for key in ('initial_rules','operation_trace','request','gold_decision','gold_state','supporting_rule_ids',
            'supporting_turn_ids','evidence_trace','ambiguity_flags','annotation_status','engine_version','relevance_mask'):
        if record.get(key)!=oracle.get(key):
            raise ValueError('guarded_oracle_replay_changed:'+key)
    return proof
