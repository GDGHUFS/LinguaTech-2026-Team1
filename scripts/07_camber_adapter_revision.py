"""CPU-only r3 audit. Writes a new revision; never generates approvals or dialogues.

Ten-record decisions are an explicitly retrospective AI review packet, NOT ranking
inputs. All other groups receive the identical automatic selector and checks.
"""
import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
sys.dont_write_bytecode = True


def denied(*args, **kwargs):
    raise RuntimeError('Network forbidden during CPU-only adapter audit')


socket.create_connection = denied
socket.socket.connect = denied
socket.socket.connect_ex = denied

from src.data.annotations import propose
from src.data.canonicalize import active_setup, download_source, load_source, select_group_candidates
from src.data.camber_evidence import category, representative_key
from src.data.mutations import make_variants
from src.data.source_policy import generation_blockers
from src.data.oracle_adapter import evaluate_structured
from src.utils.reproducibility import content_hash, file_hash


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def jsonl(path, rows):
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def snapshot():
    return {p.relative_to(ROOT).as_posix():file_hash(p) for folder in ('data','outputs')
            for p in sorted((ROOT/folder).rglob('*')) if p.is_file()}


def summarize(rows):
    return dict(records=len(rows), groups=len({r['group_id'] for r in rows}),
        categories=dict(Counter(category(r) for r in rows)),
        reasons_overlapping=dict(Counter(reason for r in rows for reason in reasons(r))),
        generation_blockers_overlapping=dict(Counter(b for r in rows for b in generation_blockers(r))),
        production_variants=sum(len(make_variants(r)) for r in rows),
        human_reviewed=sum(r['human_review_status']=='reviewed' for r in rows),
        generation_ready=sum(r['generation_ready'] for r in rows),
        actual_semantic_review_groups=0)


def reasons(row):
    checks=row['source_checks']
    result=list(checks['conflicts']) + list(checks['pending'])
    result.extend(row['exclusion_reasons'])
    for key in ('subject','sender','recipient','data_type'):
        if checks['checks'][key]['status']=='unclear': result.append(key+'_literal_alignment_unclear')
    scope=checks['checks']['information_scope']['status']
    if scope!='unclear': result.append(scope)
    if checks['checks']['relationship_reference']['conflict_indicators']:
        result.append('RELATIONSHIP_REFERENCE_CONFLICT')
    return sorted(set(result))


def diagnostic(scope):
    """Independent symbolic hypotheses, never a source Sample or generator bypass."""
    rule=dict(rule_id='diagnostic_only',authorization='ALLOW',**scope)
    base=dict(initial_rules=[rule],operation_trace=[],request=deepcopy(scope))
    examples=[('BASE',deepcopy(base))]
    for i in range(1,10):
        cap=f'CAP{i:02}'; item=deepcopy(base)
        op=dict(operation_id=cap,operation='NO_CHANGE',rule_id=rule['rule_id'])
        if i==1:
            item['initial_rules']=[]; op.update(operation='ADD',rule=deepcopy(rule))
        elif i==2: op['operation']='REVOKE'
        elif i==3: op.update(operation='MODIFY',changes={'authorization':'DENY'})
        elif i==4:
            item['initial_rules'][0]['authorization']='DENY'
            op.update(operation='EXCEPTION',rule_id='diagnostic_exception',parent_rule_id=rule['rule_id'],
                rule=dict(rule,rule_id='diagnostic_exception',scope={'channel':'secure_portal'}))
            item['request']['channel']='secure_portal'
        elif i==5:
            item['operation_trace'].append(dict(op,operation='REVOKE',operation_id='revoke_first'))
            op['operation']='REINSTATE'
        elif i in (7,8,9): op=None
        if i==9: item['request']['recipient']=None
        if op: item['operation_trace'].append(op)
        examples.append((cap,item))
    for field in ('recipient','purpose','data_type'):
        item=deepcopy(base)
        item['operation_trace']=[dict(operation_id='modify_'+field,operation='MODIFY',rule_id=rule['rule_id'],
            changes={field:'diagnostic_alternative_'+field})]
        examples.append(('CAP03_'+field,item))
    outputs=[]
    for name, item in examples:
        actual=evaluate_structured(item)
        expected='CLARIFY' if name=='CAP09' else 'DENY' if name in ('CAP02','CAP03') or name.startswith('CAP03_') else 'ALLOW'
        assert actual['gold_decision']==expected and actual['annotation_status']=='VALID'
        if name=='CAP09':
            assert actual['gold_state']['authorization']=='UNRESOLVED'
            assert actual['gold_state']['rule_status'] is None
            assert actual['gold_state']['scope_match']=='INCOMPLETE'
        if name in ('CAP06','CAP07','CAP08'):
            assert actual['gold_state']==outputs[0]['gold_state']
        outputs.append(dict(case=name,**{k:actual[k] for k in ('gold_decision','gold_state','supporting_turn_ids','annotation_status')}))
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'verification/camber_adapter_r3')
    args=parser.parse_args()
    if args.out.exists(): parser.error('Revision already exists; choose a new --out')
    args.out.mkdir(parents=True)
    out=args.out
    before=snapshot(); save(out/'artifacts_before.json',before)
    import yaml
    config=yaml.safe_load((ROOT/'configs/data_camber_r3.yaml').read_text(encoding='utf-8'))
    setup=active_setup(config); rows=[]
    for source,spec in setup['sources'].items():
        download_source(ROOT/config['paths']['raw']/source,spec,offline=True)
        rows.extend(load_source(ROOT/config['paths']['raw'],source,spec))
    pilot=select_group_candidates(rows,100,42)
    assert pilot==select_group_candidates(list(reversed(rows)),100,42)
    old=[json.loads(l) for l in (ROOT/'data/annotated/canonical_confaideplus_v2.jsonl').read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    assert {r['group_id'] for r in old}=={r['group_id'] for r in pilot}
    indexed={r['source_record_id']:r for r in rows}
    migrations=[]
    for legacy in old:
        current=indexed[legacy['source_record_id']]
        assert legacy['source_provenance']==current['source_provenance']
        migrations.append(dict(annotation_revision='adapter-migration-r3',seed_id=legacy['seed_id'],
            record_hash=current['source_provenance']['record_hash'],original_canonical_hash=content_hash(legacy),
            derived_canonical_hash=content_hash(current),human_review_status='pending',approval=None,
            changes={k:dict(before=legacy.get(k),after=current.get(k))
                for k in ('subject','sender','recipient','data_type','purpose','source_motive','disclosure_purpose','source_checks')
                if legacy.get(k)!=current.get(k)},path='scenario.reveal_reason',
            quote=ast.literal_eval(current['source_record']['scenario'])['reveal_reason'],
            reason='Adapter migration separates source motive from unconfirmed disclosure purpose. No raw or legacy canonical edit.',
            unresolved=current['source_checks']['pending']+current['source_checks']['conflicts']))
    jsonl(out/'adapter_before_after.jsonl',migrations)
    old_by_group={r['group_id']:r for r in old}
    selected={r['group_id']:r for r in pilot}
    groups=defaultdict(list)
    for row in rows: groups[row['group_id']].append(row)
    jsonl(out/'canonical_pilot_r3.jsonl',pilot)
    jsonl(out/'selection_comparison.jsonl',[
        dict(group_id=r['group_id'],previous_record_id=old_by_group[r['group_id']]['source_record_id'],
            selected_record_id=r['source_record_id'],selected_hash=r['source_provenance']['record_hash'],
            changed=r['seed_id']!=old_by_group[r['group_id']]['seed_id'],category=category(r),
            criteria_revision='camber-selection-r3',
            no_expansion_comparison=[dict(source_record_id=c['source_record_id'],
                record_hash=c['source_provenance']['record_hash'],rank=representative_key(c),reasons=reasons(c),
                information=c['data_type'],category=category(c)) for c in sorted(groups[r['group_id']],key=representative_key)
                if c['expansion']['expansion_type']=='no_expansion']) for r in pilot])
    # Explicitly retrospective AI review of named cases. This table NEVER drives selection.
    previous_actions={'222':'KEEP','141':'EXCLUDE','158':'KEEP','329':'REPLACE_EXPANSION','261':'REANNOTATE',
                      '153':'KEEP','207':'KEEP','306':'REPLACE_EXPANSION','106':'REANNOTATE','192':'REANNOTATE'}
    notes={
        '222':('they kept it from their circle of family and friends.','Past secrecy; later synthetic authority needs temporal separation.'),
        '141':('He confided this information only to Hannah','Amy in the question is not the sole confidant Hannah; do not change sender to repair story.'),
        '158':('They never discussed it with anyone else','Preserve past depression and current recovery; check whole-context CAP09.'),
        '329':('Jane has publicly shared her views','Public opinion branch differs from private infidelity; replace entire exact record, not its detail.'),
        '261':('he would be willing to bet a significant bonus on this.','Boss wager does not establish sender financial intent or disclosure purpose.'),
        '153':('They decided to keep this a secret','Historical secrecy does not grant or preclude later hypothetical ALLOW without explicit temporal boundary.'),
        '207':('His disease never became public knowledge','Preserve disease history and circle of holders; no automatic consent inference.'),
        '306':('his general views on the importance of honesty','General views branch differs from private infidelity; replacement still has unresolved purpose.'),
        '106':('Sophie is married to John','Spouse reference is Sophie; preserve John and pending wager motive/purpose.'),
        '192':("Emily's daughter, Jane",'Jane as cousin conflicts with sibling/daughter narrative; niece is an inference pending human resolution; no research consent.')}
    packet=[]; proposals=[]
    for identifier, previous in previous_actions.items():
        row=indexed['confaideplus:'+identifier]
        quote, reason=notes[identifier]
        assert quote in row['policy_context']
        chosen=selected[row['group_id']]
        legacy=next(r for r in old if r['source_record_id']==row['source_record_id'])
        conflicts=row['source_checks']['conflicts']
        action='EXCLUDE' if conflicts else 'REPLACE_EXPANSION' if chosen['seed_id']!=row['seed_id'] else previous
        packet.append(dict(source_record_id=row['source_record_id'],group_id=row['group_id'],record_hash=row['source_provenance']['record_hash'],
            previous_judgment=previous,current_ai_judgment=action,selected_record_id=chosen['source_record_id'],
            old_purpose=legacy['purpose'],new_purpose=row['purpose'],source_motive=row['source_motive'],
            source_path='scenario.story',source_quote=quote,reason=reason,automatic_category=category(row),
            source_checks=row['source_checks'],review_status='AI_preliminary_not_human_approval',
            unresolved=reasons(row),generation_ready=False))
    jsonl(out/'ten_exact_record_review.jsonl',packet)
    # Save purpose-null proposals for all 100 plus reviewed old exact records no longer selected.
    annotation_rows={r['seed_id']:r for r in pilot}
    annotation_rows.update({indexed['confaideplus:'+i]['seed_id']:indexed['confaideplus:'+i] for i in previous_actions})
    for row in annotation_rows.values():
        changes={'purpose':dict(before=row['purpose'],after=None,status='unclear',path='scenario.story',
            quote=row['source_question'],reason='No confirmed disclosure purpose; preserve motive separately.')}
        if row['source_record_id']=='confaideplus:106':
            changes['relationship_reference']=dict(before=None,after='Sophie',status='pending',path='scenario.story',
                quote='Sophie is married to John',reason='Subject-relative spouse reference; requires human confirmation.')
        if row['source_record_id']=='confaideplus:192':
            changes['recipient_relation']=dict(before=None,after='niece',status='pending',path='scenario.story',
                quote="Emily's daughter, Jane",reason='Inference conflicts with structured cousin; human conflict resolution required.')
        item=propose(row,'source-semantic-r3',changes,'CPU evidence proposal only; never semantic approval.',
            proposed_by='AI_preliminary',unresolved=reasons(row))
        proposals.append(item)
    jsonl(out/'annotation_proposals.jsonl',proposals)
    jsonl(out/'annotation_parents.jsonl',list(annotation_rows.values()))
    # New rules are also run on groups outside both the original 10 and pilot100.
    all_selected=select_group_candidates(rows,270,42)
    ten_groups={r['group_id'] for r in packet}
    pilot_groups={r['group_id'] for r in pilot}
    other90=[r for r in pilot if r['group_id'] not in ten_groups]
    independent170=[r for r in all_selected if r['group_id'] not in pilot_groups]
    jsonl(out/'independent_nonpilot_selection.jsonl',[
        dict(group_id=r['group_id'],source_record_id=r['source_record_id'],record_hash=r['source_provenance']['record_hash'],
             category=category(r),reasons=reasons(r)) for r in independent170])
    hypotheses=[]
    for identifier in list(previous_actions)+['59','36']:
        row=indexed['confaideplus:'+identifier]
        # Purpose here is an explicitly isolated HYPOTHESIS from the old mapping,
        # never inserted into an annotation or production source record.
        scope={k:row[k] for k in ('subject','sender','recipient','data_type')}
        scope['purpose']=row['source_motive']['value']
        hypotheses.append(dict(source_record_id=row['source_record_id'],record_hash=row['source_provenance']['record_hash'],
            purpose_hypothesis_only=scope['purpose'],results=diagnostic(scope)))
    save(out/'symbolic_diagnostic_summary.json',dict(kind='isolated_symbolic_hypothesis_not_source_sample',
        cases=156,source_scopes=12,passed=156,failed=0,natural_language_validated=0,hypotheses=hypotheses))
    after=snapshot()
    integrity=dict(files_before=len(before),files_after=len(after),
        changed=[p for p in before if after.get(p)!=before[p]],added=sorted(set(after)-set(before)),
        missing=sorted(set(before)-set(after)))
    assert not integrity['changed'] and not integrity['added'] and not integrity['missing']
    save(out/'artifact_integrity.json',integrity)
    result=dict(revision='camber-adapter-r3',source_selection=config['source_selection'],
        source_integrity='PASS_pinned_git_blob_hash',raw_records=len(rows),upstream_groups=len(groups),
        pilot=summarize(pilot),other90=summarize(other90),independent170=summarize(independent170),
        no_expansion_in_pilot_groups=summarize([r for r in rows if r['group_id'] in pilot_groups and r['expansion']['expansion_type']=='no_expansion']),
        selected_groups_unchanged=True,reproducible_under_reversed_input=True,
        representative_changes=sum(r['seed_id']!=old_by_group[r['group_id']]['seed_id'] for r in pilot),
        ten_ai_judgments=dict(Counter(r['current_ai_judgment'] for r in packet)),
        automatic_semantic_approval=False,actual_source_reading='10 legacy exact records + two replacement candidates; AI only',
        fully_semantically_validated_pilot_groups=0,annotation_proposals=len(proposals),annotation_approvals=0,
        source_approvals=0,production_approvals=0,qwen_calls=0,target_calls=0,model_downloads=0,
        training='NOT_RUN',split_creation='NOT_RUN',selection_retrospective=True,
        code_config_hashes={str(p):file_hash(ROOT/p) for p in ('src/data/camber_evidence.py','src/data/canonicalize.py',
            'src/data/annotations.py','src/data/validators.py','src/data/source_policy.py','src/data/approvals.py',
            'src/data/generation.py','src/data/contracts.py','src/data/mutations.py','configs/data_camber_r3.yaml','configs/camber_selection_r3.yaml')},
        forbidden_model_modules_imported=[m for m in ('torch','transformers') if m in sys.modules])
    assert not result['forbidden_model_modules_imported']
    save(out/'summary.json',result)
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__=='__main__': main()
