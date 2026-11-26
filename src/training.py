"""CPU probes with equal L2 search budgets and validation-only selection."""
import copy
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .contracts import (Blocked, digest, file_hash, paths, read_json, read_jsonl,
                        write_json, write_jsonl, gate, code_hash)
from .checkpointing import save_torch, save_joblib
from .features import temporal_features
from .representation import SampleCache
from .probes import PROBES
from .metrics import metrics, select_threshold
from .baselines import build_baselines, baseline_input
from .training_selection import patience_improved, select_l2_candidate
from .probe_diagnostics import diagnose


def probe_diagnostics(model, train_features, train_y, val_features, val_y,
                      rows=None, threshold=.5):
    """No test inputs; subgroup cutpoints and fixed gate come from train only."""
    with torch.inference_mode():
        gate_parameters = ({'a':float(model.gate_a.item()),'c':float(model.gate_c.item())}
                           if hasattr(model,'gate_a') else None)
        fixed_gate = (float(model.gate_value(train_features[2]).mean())
                      if gate_parameters is not None else 1.)
        weight = model.head.weight.detach().cpu().numpy()
        bias = float(model.head.bias.item())
    cuts = {'distance':np.quantile(train_features[2].cpu().numpy().ravel(),[.33,.67]).tolist()}
    if rows is not None:
        cuts['turn'] = np.quantile([r['turn_index'] for r in rows[0]],[.33,.67]).tolist()
    reports, predictions = {}, {}
    for i,(split,features,y) in enumerate((('train',train_features,train_y),
                                          ('validation',val_features,val_y))):
        reports[split], predictions[split] = diagnose(
            tuple(t.detach().cpu().numpy() for t in features), y.detach().cpu().numpy(),
            weight,bias,gate_parameters,fixed_gate,threshold,
            rows=None if rows is None else rows[i],cutpoints=cuts)
    return reports, predictions


def seed_all():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)


def feature_tensors(rows, cache):
    if not rows:
        raise Blocked('Empty cohort')
    current = torch.stack([cache.vector(r['representation_key']) for r in rows])
    previous = torch.stack([cache.vector(r['previous_representation_key']) for r in rows])
    return temporal_features(current, previous)


def labels(rows):
    return torch.tensor([r['failure_label'] for r in rows], dtype=torch.float32)


def predict_probe(model, features):
    model.eval()
    with torch.inference_mode():
        return model(*features).sigmoid().cpu().numpy()


def fit_probe(name, train_features, train_y, validation_features, validation_y, cfg,
              checkpoint_dir=None, resume=False, provenance=None, diagnostic_rows=None):
    if len(torch.unique(validation_y)) != 2 or len(torch.unique(train_y)) != 2:
        raise Blocked('Train and validation require both classes')
    seed_all()
    model = PROBES[name](train_features[0].shape[-1]).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=0.)
    dataset = TensorDataset(*train_features, train_y)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True,
                        generator=generator, num_workers=0)
    best, best_state, wait, history = None, None, 0, []
    patience_reference = None
    min_delta = cfg['early_stopping_min_delta']
    patience_improved((0., 0.), None, min_delta)  # Validate before fitting.
    provenance = provenance or {}
    signature = digest({'protocol':'conshift-training-v3', 'diagnostics_version':1,
        'name':name, 'config':cfg, 'provenance':provenance,
        'tensors':[{'shape':list(t.shape), 'sha256':hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
                   for t in (*train_features,train_y,*validation_features,validation_y)]})
    last_path = Path(checkpoint_dir) / (name + '.last.pt') if checkpoint_dir is not None else None
    best_path = Path(checkpoint_dir) / (name + '.best.pt') if checkpoint_dir is not None else None
    start_epoch, elapsed_before, finished = 1, 0., False
    if last_path is not None and last_path.exists():
        if not resume:
            raise Blocked(f'{name} has a saved run; use --resume')
        state = torch.load(last_path,map_location='cpu',weights_only=True)
        if state['signature'] != signature:
            raise Blocked('Training checkpoint configuration/cohort/features mismatch')
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer_state'])
        generator.set_state(state['generator_state'])
        torch.set_rng_state(state['torch_rng_state'])
        random.setstate(state['python_rng_state'])
        ns = state['numpy_rng_state']
        np.random.set_state((ns[0],np.asarray(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
        best, best_state, best_epoch = tuple(state['best_rank']),state['best_state'],state['best_epoch']
        wait, history = state['stale_epochs'],state['history']
        patience_reference = tuple(state['patience_reference'])
        start_epoch, elapsed_before, finished = state['epoch'] + 1,state['elapsed_seconds'],state['finished']
    elif best_path is not None and best_path.exists():
        raise Blocked('Best checkpoint exists without resumable last checkpoint')
    started = time.perf_counter()
    epochs = range(start_epoch, cfg['max_epochs'] + 1) if not finished else []
    for epoch in epochs:
        model.train()
        losses = []
        for u, delta, distance, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(u, delta, distance), y)
            loss = loss + cfg['l2_lambda'] * model.head.weight.square().sum()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            train_logits = model(*train_features)
            validation_logits = model(*validation_features)
            train_bce = float(F.binary_cross_entropy_with_logits(train_logits, train_y))
            validation_bce = float(F.binary_cross_entropy_with_logits(validation_logits, validation_y))
            train_probabilities = train_logits.sigmoid().cpu().numpy()
            probabilities = validation_logits.sigmoid().cpu().numpy()
        train_result = metrics(train_y.numpy(), train_probabilities, .5)
        result = metrics(validation_y.numpy(), probabilities, .5)
        rank = (result['AP'], -result['Brier'])
        history.append({'epoch': epoch, 'mean_batch_loss': float(np.mean(losses)),
                        'train_bce': train_bce, 'validation_bce': validation_bce,
                        'train_ap': train_result['AP'], 'validation_ap': result['AP'],
                        'train_brier': train_result['Brier'], 'validation_brier': result['Brier'],
                        'train_minus_validation_ap': train_result['AP'] - result['AP'],
                        'validation_minus_train_bce': validation_bce - train_bce,
                        'validation_minus_train_brier': result['Brier'] - train_result['Brier']})
        if best is None or rank > best:
            best, best_state, best_epoch = rank, copy.deepcopy(model.state_dict()), epoch
            improved = True
        else:
            improved = False
        reset_patience = patience_improved(rank, patience_reference, min_delta)
        if reset_patience:
            patience_reference, wait = rank, 0
        else:
            wait += 1
        history[-1].update(best_checkpoint_improved=improved,
                           patience_reset=reset_patience, stale_epochs=wait)
        diagnostics, _ = probe_diagnostics(model,train_features,train_y,
            validation_features,validation_y,rows=diagnostic_rows)
        history[-1]['diagnostics'] = diagnostics
        finished = wait >= cfg['patience'] or epoch == cfg['max_epochs']
        elapsed = elapsed_before + time.perf_counter() - started
        if last_path is not None:
            ns = np.random.get_state()
            state = {'schema':'conshift-training-v3','signature':signature,'epoch':epoch,
                'state_dict':copy.deepcopy(model.state_dict()),'optimizer_state':optimizer.state_dict(),
                'generator_state':generator.get_state(),'torch_rng_state':torch.get_rng_state(),
                'python_rng_state':random.getstate(),
                'numpy_rng_state':(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]),
                'best_rank':best,'best_state':best_state,'best_epoch':best_epoch,
                'stale_epochs':wait,'patience_reference':patience_reference,
                'history':history,'elapsed_seconds':elapsed,'finished':finished}
            epoch_checkpoint = {'state_dict':copy.deepcopy(model.state_dict()),'metadata':{
                **provenance,'model_name':name,'hidden_dim':train_features[0].shape[-1],
                'threshold':select_threshold(validation_y.numpy(),probabilities),
                'threshold_source':'validation','epoch':epoch,'validation_ap':rank[0],
                'validation_brier':-rank[1],'training_config':cfg,'run_signature':signature,
                'parameter_count':sum(p.numel() for p in model.parameters())}}
            save_torch(Path(checkpoint_dir) / 'epochs' / f'{name}.epoch-{epoch:03d}.pt', epoch_checkpoint)
            save_torch(last_path,state)
            if improved:
                save_torch(best_path,{'state_dict':best_state,'metadata':{
                    **provenance,'model_name':name,'hidden_dim':train_features[0].shape[-1],
                    'threshold':select_threshold(validation_y.numpy(),probabilities),
                    'threshold_source':'validation','epoch':epoch,'validation_ap':best[0],
                    'validation_brier':-best[1],'training_config':cfg,'run_signature':signature,
                    'parameter_count':sum(p.numel() for p in model.parameters())}})
            write_jsonl(Path(checkpoint_dir).parent / 'logs' / (name + '_training.jsonl'),history)
            print(json.dumps({'model':name,'epoch':epoch,'train_ap':train_result['AP'],
                'validation_ap':result['AP'],'validation_bce':validation_bce,
                'validation_brier':result['Brier'],'validation_gate':diagnostics['validation']['gate'],
                'gate_parameters':diagnostics['validation']['gate_parameters'],
                'stale_epochs':wait}),flush=True)
        if finished:
            break
    model.load_state_dict(best_state)
    model.eval()
    threshold = select_threshold(validation_y.numpy(), predict_probe(model, validation_features))
    metadata = {**provenance, 'model_name': name, 'hidden_dim': train_features[0].shape[-1],
                'threshold': threshold, 'threshold_source': 'validation', 'epoch': best_epoch,
                'validation_ap': best[0], 'validation_brier': -best[1],
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'training_seconds': elapsed_before + time.perf_counter() - started,
                'training_config': cfg,'run_signature':signature,
                'epochs_completed':len(history),'early_stopped':len(history) < cfg['max_epochs']}
    if best_path is not None:
        save_torch(best_path,{'state_dict':model.state_dict(),'metadata':metadata})
    return model, metadata, history


def load_probe(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    meta = checkpoint['metadata']
    if meta['threshold_source'] != 'validation':
        raise Blocked('Checkpoint lacks fixed validation threshold')
    model = PROBES[meta['model_name']](meta['hidden_dim'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval(), meta


def search_probe(name, train_features, train_y, val_features, val_y, cfg,
                 checkpoint_dir, resume=False, provenance=None, diagnostic_rows=None):
    """Fit each candidate once from the same seed; never inspect test data."""
    candidates = []
    root = Path(checkpoint_dir) / 'search' / name
    for index, coefficient in enumerate(cfg['l2_candidates']):
        candidate_cfg = {**cfg, 'l2_lambda':coefficient}
        folder = root / f'l2-{index:02d}' / 'checkpoints'
        _, meta, _ = fit_probe(name,train_features,train_y,val_features,val_y,
            candidate_cfg,checkpoint_dir=folder,resume=resume,provenance=provenance,
            diagnostic_rows=diagnostic_rows)
        candidates.append({**meta,'candidate_index':index,'l2_lambda':coefficient,
                           'checkpoint':str((folder / (name + '.best.pt')).resolve())})
        write_json(root / 'progress.json',{'completed_candidates':candidates,
                   'selection_source':'validation','search_config':cfg})
    selected = candidates[select_l2_candidate(candidates)]
    model, meta = load_probe(selected['checkpoint'])
    meta.update(search_config=cfg, selected_l2=selected['l2_lambda'],
                hyperparameter_selection_source='validation',
                search_training_seconds=sum(c['training_seconds'] for c in candidates))
    report = {'model_name':name,'selection_source':'validation',
              'selection_rule':'AP max, Brier min, earlier epoch, grid order',
              'candidates':candidates,'selected_l2':meta['selected_l2'],
              'selected_candidate_index':selected['candidate_index']}
    return model, meta, report


def train(cfg, resume=False):
    data, output = paths(cfg)
    collection_output = Path(cfg['paths'].get('collection_output',output))
    coverage = read_json(collection_output / 'reports/label_coverage.json')
    if not coverage['all_original_turns_collected']:
        raise Blocked('G3 final collection/coverage gate did not pass')
    labeled = read_jsonl(data / 'labeled/turns.jsonl')
    if digest(labeled) != coverage['labeled_hash']:
        raise Blocked('Labeled data changed after coverage audit')
    from .labeling import coverage as label_coverage
    coverage = {**coverage, **label_coverage(labeled)}
    if coverage['status'] != 'PASS':
        raise Blocked('Train, validation and ID test must each contain both classes')
    records = [r for r in labeled if r['eligible']]
    if digest([r['sample_id'] for r in records]) != coverage['cohort_hash']:
        raise Blocked('Cohort changed after coverage audit')
    train_rows = [r for r in records if r['split'] == 'train']
    val_rows = [r for r in records if r['split'] == 'validation']
    checkpoint_dir = output / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if any(p.name != '.gitkeep' for p in checkpoint_dir.iterdir()) and not resume:
        raise Blocked('Training artifacts already exist; use --resume to continue the same run')
    cache = SampleCache(data)
    run = read_json(data / 'manifests/collection_lock.json')
    if run['target'] != cfg['target']:
        raise Blocked('Cached Target configuration differs from training config')
    for r in records:
        cache.load(r['sample_id'], run['fingerprint'], r['prompt_hash'], r['prompt_token_ids_hash'])
    train_features, val_features = feature_tensors(train_rows, cache), feature_tensors(val_rows, cache)
    provenance = {'cohort_hash': coverage['cohort_hash'], 'collection_fingerprint': run['fingerprint'],
                  'labeled_hash': coverage['labeled_hash'],
                  'representation_protocol': {'dtype':run['dtype'], 'template_hash':run['template'],
                      'code_hash':run['code'], 'versions':run['versions']},
                  'split_hash': read_json(data / 'manifests/split_manifest.json')['manifest_hash'],
                  'target_config': cfg['target'],
                  'train_sample_hash': digest([r['sample_id'] for r in train_rows]),
                  'validation_sample_hash': digest([r['sample_id'] for r in val_rows])}
    summary = {}
    training_lock = {'provenance':provenance,'training_config':cfg['training'],
                     'protocol':'validation-l2-search-v1','training_code_hash':code_hash()}
    write_json(checkpoint_dir / 'training_run.json',training_lock,immutable=True)
    for name in PROBES:
        deployed = checkpoint_dir / (name + '.pt')
        if resume and deployed.exists():
            model, meta = load_probe(deployed)
            if (meta['model_name'] != name or any(meta.get(k) != v for k,v in provenance.items())
                    or meta.get('search_config') != cfg['training']):
                raise Blocked('Completed probe checkpoint differs from the resumed run')
        else:
            model, meta, search_report = search_probe(name, train_features, labels(train_rows), val_features, labels(val_rows),
                cfg['training'],checkpoint_dir=checkpoint_dir,resume=resume,provenance=provenance,
                diagnostic_rows=(train_rows,val_rows))
            write_json(output / 'reports' / (name + '_l2_search.json'),search_report)
            save_torch(deployed,{'state_dict':model.state_dict(),'metadata':meta})
        # Keep a convenient top-level log for the selected candidate.
        selected_index = cfg['training']['l2_candidates'].index(meta['selected_l2'])
        selected_log = checkpoint_dir / 'search' / name / f'l2-{selected_index:02d}' / 'logs' / (name + '_training.jsonl')
        write_jsonl(output / 'logs' / (name + '_training.jsonl'),read_jsonl(selected_log))
        diagnostics, diagnostic_predictions = probe_diagnostics(model,train_features,labels(train_rows),
            val_features,labels(val_rows),rows=(train_rows,val_rows),threshold=meta['threshold'])
        write_json(output / 'reports' / (name + '_diagnostics.json'),
                   {'selected_epoch':meta['epoch'],'threshold_source':'validation',
                    'provenance':provenance,'splits':diagnostics})
        for split, predictions in diagnostic_predictions.items():
            write_jsonl(output / 'diagnostics' / (name + '_' + split + '.jsonl'),predictions)
        summary[name] = meta
        write_json(output / 'reports/training_progress.json',{'completed_models':list(summary),'run':training_lock})
    for name, model in build_baselines().items():
        artifact, metadata_path = checkpoint_dir / (name + '.joblib'),checkpoint_dir / (name + '.json')
        if resume and artifact.exists() and metadata_path.exists():
            meta = read_json(metadata_path)
            if any(meta.get(k) != v for k,v in provenance.items()) or meta['artifact_hash'] != file_hash(artifact):
                raise Blocked('Completed baseline checkpoint differs from the resumed run')
            summary[name] = meta
            continue
        started = time.perf_counter()
        model.fit(baseline_input(name, train_rows), labels(train_rows).numpy())
        probabilities = model.predict_proba(baseline_input(name, val_rows))[:, 1]
        meta = {**provenance, 'model_name': name, 'threshold_source': 'validation',
                'threshold': select_threshold(labels(val_rows).numpy(), probabilities),
                'training_seconds': time.perf_counter() - started,
                'parameter_count': int(model[-1].coef_.size + model[-1].intercept_.size)}
        save_joblib(artifact,model)
        meta['artifact_hash'] = file_hash(artifact)
        write_json(metadata_path,meta)
        summary[name] = meta
        write_json(output / 'reports/training_progress.json',{'completed_models':list(summary),'run':training_lock})
    quantiles = {'turn': np.quantile([r['turn_index'] for r in train_rows], [.33, .67]).tolist(),
                 'distance': np.quantile(train_features[2].numpy().ravel(), [.33, .67]).tolist(),
                 'fit_split': 'train', **provenance}
    write_json(checkpoint_dir / 'analysis_cutpoints.json', quantiles)
    write_json(output / 'reports/training_manifest.json', {'models': summary, **provenance})
    gate(cfg, 'G4', 'PASS', same_cohort=True, target_gradients=False,
         train_only_fitting=True, checkpoint_selection='validation AP/Brier/earliest epoch')
    return summary
