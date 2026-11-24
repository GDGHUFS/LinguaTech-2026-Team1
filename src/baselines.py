"""Fixed G1 and L1 controls fit exclusively on the common train cohort."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def length_features(rows):
    return np.log1p(np.array([[r['turn_index'], r['prompt_token_count']] for r in rows], dtype=float))


def baseline_input(name, rows):
    if name == 'g1':
        return [r['prompt_text'] for r in rows]
    if name == 'l1':
        return length_features(rows)
    raise ValueError(name)


def build_baselines():
    def lr():
        return LogisticRegression(C=1., solver='liblinear', max_iter=1000,
                                  random_state=42, class_weight=None)
    return {'g1': make_pipeline(TfidfVectorizer(ngram_range=(1, 2), lowercase=True,
                max_features=20000, min_df=2, sublinear_tf=True, norm='l2'), lr()),
            'l1': make_pipeline(StandardScaler(), lr())}
