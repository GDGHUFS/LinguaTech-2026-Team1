"""실험 seed를 검증하고 불확실성 추정의 범위를 기록한다."""


def seed_metadata(seeds):
    if (not isinstance(seeds, (list, tuple)) or not seeds or
            any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError('Nonempty distinct integer training seeds required')
    return dict(training_seeds=list(seeds), training_seed_count=len(seeds),
                single_seed_experiment=len(seeds) == 1,
                seed_variability_estimated=len(seeds) > 1,
                uncertainty_scope='scenario_groups_conditional_on_registered_training_seeds')
