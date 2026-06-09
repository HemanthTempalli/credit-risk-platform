"""
src/features.py
Domain-driven feature engineering for credit risk scoring.
"""

import pandas as pd
import numpy as np


EXT_WEIGHTS = {'EXT_SOURCE_1': 0.2, 'EXT_SOURCE_2': 0.5, 'EXT_SOURCE_3': 0.3}

FEATURE_COLS = [
    # Original numerical
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'REGION_RATING_CLIENT', 'HOUR_APPR_PROCESS_START', 'CNT_CHILDREN',
    'FLAG_DOCUMENT_3', 'AMT_REQ_CREDIT_BUREAU_YEAR',
    # Engineered
    'AGE_YEARS', 'YEARS_EMPLOYED', 'EMPLOYMENT_RATIO',
    'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'CREDIT_TERM',
    'GOODS_PRICE_RATIO', 'INCOME_PER_PERSON',
    'EXT_SOURCE_MEAN', 'EXT_SOURCE_MIN', 'EXT_SOURCE_STD',
    'EXT_SOURCE_PROD', 'EXT_WEIGHTED',
    'HIGH_CREDIT_STRESS', 'UNEMPLOYED_FLAG', 'YOUNG_BORROWER',
    'HIGH_ANNUITY_RATIO', 'SOCIAL_DEFAULT_RATIO',
    'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CODE_GENDER',
    'EDUCATION_ORDINAL', 'IS_REVOLVING',
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-driven feature engineering to raw Home Credit data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with Home Credit schema columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with original + engineered columns.
    """
    df = df.copy()

    # ── Temporal features ──────────────────────────────────────────────────────
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    df['YEARS_EMPLOYED'] = np.where(
        df['DAYS_EMPLOYED'] == 365243, 0.0,
        -df['DAYS_EMPLOYED'] / 365
    )
    df['EMPLOYMENT_RATIO'] = df['YEARS_EMPLOYED'] / (df['AGE_YEARS'] + 1e-6)
    df['YEARS_REGISTRATION'] = -df['DAYS_REGISTRATION'] / 365
    df['YEARS_ID_PUBLISH'] = -df['DAYS_ID_PUBLISH'] / 365

    # ── Credit stress ratios ───────────────────────────────────────────────────
    df['CREDIT_INCOME_RATIO']  = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['CREDIT_TERM']          = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)
    df['GOODS_PRICE_RATIO']    = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + 1)
    df['INCOME_PER_PERSON']    = df['AMT_INCOME_TOTAL'] / (df['CNT_CHILDREN'] + 1)

    # ── External score aggregations ────────────────────────────────────────────
    ext_cols = [c for c in EXT_WEIGHTS if c in df.columns]
    if ext_cols:
        df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
        df['EXT_SOURCE_MIN']  = df[ext_cols].min(axis=1)
        df['EXT_SOURCE_STD']  = df[ext_cols].std(axis=1).fillna(0)
        df['EXT_SOURCE_PROD'] = df[ext_cols].prod(axis=1)
        df['EXT_WEIGHTED'] = sum(
            df[c].fillna(df[c].median()) * w
            for c, w in EXT_WEIGHTS.items() if c in df.columns
        )

    # ── Risk flags ─────────────────────────────────────────────────────────────
    df['HIGH_CREDIT_STRESS'] = (df['CREDIT_INCOME_RATIO'] > 5).astype(int)
    df['UNEMPLOYED_FLAG']    = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['YOUNG_BORROWER']     = (df['AGE_YEARS'] < 27).astype(int)
    df['HIGH_ANNUITY_RATIO'] = (df['ANNUITY_INCOME_RATIO'] > 0.3).astype(int)

    # ── Social circle risk ─────────────────────────────────────────────────────
    if 'OBS_30_CNT_SOCIAL_CIRCLE' in df.columns and 'DEF_30_CNT_SOCIAL_CIRCLE' in df.columns:
        df['SOCIAL_DEFAULT_RATIO'] = (
            df['DEF_30_CNT_SOCIAL_CIRCLE'] /
            (df['OBS_30_CNT_SOCIAL_CIRCLE'] + 1)
        )

    # ── Categorical encoding ───────────────────────────────────────────────────
    binary_map = {'Y': 1, 'N': 0, 'M': 1, 'F': 0, 'XNA': -1}
    for col in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CODE_GENDER']:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0)

    edu_order = {
        'Lower secondary': 0,
        'Secondary / secondary special': 1,
        'Incomplete higher': 2,
        'Higher education': 3,
        'Academic degree': 4,
    }
    if 'NAME_EDUCATION_TYPE' in df.columns:
        df['EDUCATION_ORDINAL'] = df['NAME_EDUCATION_TYPE'].map(edu_order).fillna(1)

    if 'NAME_CONTRACT_TYPE' in df.columns:
        df['IS_REVOLVING'] = (df['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype(int)

    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the final feature matrix (subset of engineered df)."""
    df_eng = engineer_features(df)
    cols = [c for c in FEATURE_COLS if c in df_eng.columns]
    return df_eng[cols]
