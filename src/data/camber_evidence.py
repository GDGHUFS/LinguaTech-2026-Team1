"""Conservative, versioned source checks. Pattern evidence is never semantic approval."""
import ast
import re

REVISION = 'camber-evidence-r3'


def assess(seed):
    scenario = ast.literal_eval(seed['source_record']['scenario'])
    story = scenario.get('story', '')
    narrative, _, tail = story.partition('Question:')
    question = tail.partition('Answer:')[0].strip()
    checks, conflicts, pending = {}, [], []
    pattern = r"Is it ok for (?P<sender>.+?) to share information about (?P<object>.+?) with (?P<recipient>.+?)\?"
    match = re.search(pattern, question)
    for field in ('subject', 'sender', 'recipient', 'data_type'):
        value = seed.get(field)
        quote = None
        status = 'unclear'
        if match and value:
            fragment = match['object'] if field in ('subject', 'data_type') else match[field]
            quote = fragment
            # Synonyms and relational phrases require semantic review, never automatic mismatch.
            if re.search(r'(?<!\w)' + re.escape(value) + r'(?!\w)', fragment, re.I):
                status = 'literal_match'
            elif field == 'sender' and re.fullmatch(r'[A-Z][a-z]+', value) and re.fullmatch(r'[A-Z][a-z]+', fragment):
                status = 'conflict'
                conflicts.append('QUESTION_SENDER_CONFLICT')
            elif field == 'recipient' and re.fullmatch(r'[A-Z][a-z]+', value) and re.search(r'\b[A-Z][a-z]+$', fragment):
                status = 'conflict'
                conflicts.append('QUESTION_RECIPIENT_CONFLICT')
            elif field == 'subject' and re.fullmatch(r'[A-Z][a-z]+', value) and re.search(r"\b[A-Z][a-z]+['’]s\b", fragment):
                status = 'conflict'
                conflicts.append('QUESTION_SUBJECT_CONFLICT')
        checks[field] = dict(status=status, path='scenario.story', quote=quote)
        if status == 'unclear':
            pending.append(field + '_question_alignment')
    # Restrict exclusive-confidant conflict to short focal detail; a sibling's added
    # fact can have a different holder. Long/mixed descriptions remain unresolved.
    holders = list(re.finditer(r'confided this information only to ([A-Z][a-z]+)', narrative))
    holder_conflict = bool(holders and re.fullmatch(r'[A-Z][a-z]+', seed.get('sender') or '')
                           and len((seed.get('data_type') or '').split()) <= 6
                           and any(m[1] != seed.get('sender') for m in holders))
    if holder_conflict:
        conflicts.append('SOURCE_KNOWLEDGE_CONFLICT')
    checks['knowledge_holder'] = dict(status='conflict' if holder_conflict else 'unclear',
        path='scenario.story', quote=holders[0][0] if holders else None)
    pending.append('knowledge_holder_semantic_review')
    # Publicness must concern the requested information, not another fact in story.
    scope_text = ((seed.get('data_type') or '') + ' ' + (match['object'] if match else '')).lower()
    public = bool(re.search(r'\b(public|publicly|published)\b', scope_text))
    general = bool(re.search(r'\b(general (views|information|report)|views on|support groups)\b', scope_text))
    checks['information_scope'] = dict(status='public_indicator' if public else
        'general_information_indicator' if general else 'unclear',
        path='scenario.detail', quote=seed.get('data_type'))
    checks['relationship_reference'] = dict(status='unclear',
        path='scenario.story', quote=question,
        structured_relations={k: scenario.get(k) for k in ('aware_agent_relation', 'oblivious_agent_relation')},
        reason='Possessive their may use sender or subject as reference; inspect narrative kinship.')
    relation_conflicts = []
    sibling_sentence = re.search(r'([A-Z][a-z]+(?:,? (?:and )?[A-Z][a-z]+)+) are siblings', narrative)
    if sibling_sentence:
        siblings = set(re.findall(r'[A-Z][a-z]+', sibling_sentence[1]))
        for kin in re.finditer(r"([A-Z][a-z]+)'s daughter, ([A-Z][a-z]+)", narrative):
            if (kin[1] in siblings and seed.get('sender') in siblings and kin[2] == seed.get('recipient')
                    and scenario.get('oblivious_agent_relation') == 'cousin'):
                relation_conflicts.append(dict(code='RELATIONSHIP_REFERENCE_CONFLICT', path='scenario.story', quote=kin[0]))
    checks['relationship_reference']['conflict_indicators'] = relation_conflicts
    pending.extend(['relationship_reference', 'information_publicness', 'disclosure_purpose',
                    'historical_secrecy_vs_later_synthetic_policy'])
    return dict(revision=REVISION, checks=checks, conflicts=sorted(set(conflicts)),
                pending=sorted(set(pending)), semantic_status='not_validated')


def category(seed):
    if seed.get('exclusion_reasons') or seed.get('source_checks', {}).get('conflicts'):
        return 'excluded'
    checks = seed.get('source_checks', {}).get('checks', {})
    if checks.get('relationship_reference', {}).get('conflict_indicators'):
        return 'uncertain'
    if any(checks.get(k, {}).get('status') != 'literal_match' for k in ('subject', 'sender', 'recipient', 'data_type')):
        return 'uncertain'
    if checks.get('information_scope', {}).get('status') != 'unclear':
        return 'uncertain'
    # Structural candidate only: pending purpose/publicness/relations still block generation.
    return 'structural_candidate'


def representative_key(row):
    checks = row.get('source_checks', {}).get('checks', {})
    category_rank = {'structural_candidate': 0, 'uncertain': 1, 'excluded': 2}[category(row)]
    scope_rank = {'unclear': 0, 'general_information_indicator': 1, 'public_indicator': 2}.get(
        checks.get('information_scope', {}).get('status'), 1)
    actor_uncertainty = sum(checks.get(k, {}).get('status') != 'literal_match' for k in ('subject', 'sender', 'recipient'))
    purpose_pending = row.get('disclosure_purpose', {}).get('status') != 'supported'
    identifier = row['source_record_id']
    suffix = identifier.rsplit(':', 1)[-1]
    return (category_rank, scope_rank, actor_uncertainty, purpose_pending,
            len((row.get('data_type') or '').split()),
            (0, int(suffix)) if suffix.isdecimal() else (1, suffix), identifier)
