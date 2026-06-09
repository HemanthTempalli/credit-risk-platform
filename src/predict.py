"""
src/predict.py
Inference utilities: load artifacts, score applicants, explain predictions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from features import get_feature_matrix


ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')


def load_pipeline() -> Tuple:
    """Load calibrated model, preprocessor, and metadata."""
    model      = joblib.load(os.path.join(ARTIFACT_DIR, 'model_calibrated.pkl'))
    preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, 'preprocessor.pkl'))
    with open(os.path.join(ARTIFACT_DIR, 'metadata.json')) as f:
        metadata = json.load(f)
    return model, preprocessor, metadata


def predict_default_probability(
    raw_row: Dict,
    model,
    preprocessor,
    metadata: Dict
) -> Dict:
    """
    Score a single applicant.

    Parameters
    ----------
    raw_row : dict
        Raw feature dictionary (Home Credit schema).
    model, preprocessor, metadata : loaded artifacts

    Returns
    -------
    dict with keys:
        prob          – calibrated default probability
        decision      – 'APPROVED' | 'MANUAL REVIEW' | 'DECLINED'
        threshold     – decision threshold used
        feature_names – list of features fed to model
    """
    df = pd.DataFrame([raw_row])
    X  = get_feature_matrix(df)
    X_proc = preprocessor.transform(X)

    prob = float(model.predict_proba(X_proc)[0, 1])
    threshold = metadata.get('optimal_threshold', 0.35)

    if prob < threshold * 0.6:
        decision = 'APPROVED'
    elif prob < threshold:
        decision = 'MANUAL REVIEW'
    else:
        decision = 'DECLINED'

    return {
        'prob': prob,
        'decision': decision,
        'threshold': threshold,
        'feature_names': X.columns.tolist(),
        'feature_values': X.iloc[0].to_dict(),
    }
