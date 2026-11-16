"""검증 지표로 중단 시점과 학습 요약을 관리한다. 모델 실행 환경은 필요하지 않다."""
import json
from pathlib import Path

CORE_VARIANTS = ('A', 'B_pred', 'C', 'C_plus', 'C+')
COMPARISON_NOTE = ('Early stopping can produce different actual optimizer update steps across models; '
                   'this is not the specification\'s equal-update-budget comparison.')


def checkpoint_is_better(metrics, best):
    return best is None or (metrics['validation_auprc'], -metrics['validation_brier']) > (
        best['validation_auprc'], -best['validation_brier'])


def update_early_stopping(state, metrics, patience):
    """AUPRC가 엄격히 개선될 때만 patience를 초기화한다. Brier는 checkpoint 동률 해소에만 쓴다."""
    previous = state.get('early_stopping_best_auprc')
    if previous is None or metrics['validation_auprc'] > previous:
        state['early_stopping_best_auprc'] = metrics['validation_auprc']
        state['epochs_without_auprc_improvement'] = 0
    else:
        state['epochs_without_auprc_improvement'] = state.get('epochs_without_auprc_improvement', 0) + 1
    state['early_stopped'] = state['epochs_without_auprc_improvement'] >= patience
    return state['early_stopped']


def training_summary(state, metadata, config):
    best = state.get('best') or {}
    return dict(model=metadata['mode'], seed=metadata['seed'], trial=metadata.get('trial', 0),
                data_kind=metadata['data_kind'], max_epochs=config['max_epochs'],
                completed_epochs=state['epoch'],
                epochs_started=state['epoch'] + int(state['next_batch'] > 0),
                actual_optimizer_steps=state['global_step'],
                optimizer_step_budget=metadata['optimizer_step_budget'],
                best_epoch=best.get('epoch'), best_optimizer_step=best.get('step'),
                validation_auprc=best.get('validation_auprc'), validation_brier=best.get('validation_brier'),
                training_time_seconds=state['runtime_seconds'],
                training_time_scope='training_and_validation_excludes_model_loading_and_checkpoint_io',
                stop_reason=state.get('stop_reason'), early_stopped=state.get('early_stopped', False),
                epochs_without_auprc_improvement=state.get('epochs_without_auprc_improvement', 0),
                early_stopping=config.get('early_stopping'),
                comparison_note=COMPARISON_NOTE,
                common_initial_settings={k: config[k] for k in (
                    'optimizer', 'learning_rate', 'weight_decay', 'scheduler', 'per_device_batch_size',
                    'gradient_accumulation_steps', 'effective_batch_size', 'max_grad_norm')},
                loss_weights={k: config[k] for k in ('lambda_state', 'mu_relevance', 'nu_type')},
                train_hash=metadata.get('train_hash'), validation_hash=metadata.get('validation_hash'))


def write_training_summary(directory, state, metadata, config):
    summary = training_summary(state, metadata, config)
    (Path(directory)/'training_summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return summary


def write_training_comparison(root):
    root = Path(root)
    runs = []
    for path in sorted(root.glob('*/seed_42/trial_*/training_summary.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        if row['model'] not in CORE_VARIANTS:
            continue
        runs.append(dict(row, summary_path=str(path)))
    report = dict(training_seeds=[42], single_seed_experiment=True, runs=runs,
                  comparison_note=COMPARISON_NOTE,
                  specification_deviations=['maximum_epochs_3_to_10', 'equal_actual_update_budget_not_enforced',
                                            'three_training_seeds_to_seed_42_only'],
                  warning='Check dataset hashes and initial settings before comparing runs from different experiment revisions.')
    (root/'training_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report
