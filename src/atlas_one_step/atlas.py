from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_curve, roc_auc_score, auc, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_summaries(sweep_dir: str | Path) -> pd.DataFrame:
    paths = sorted(Path(sweep_dir).rglob('summary.json'))
    if not paths:
        raise FileNotFoundError(f'No summary.json files found under {sweep_dir}')
    rows = []
    for p in paths:
        with p.open('r', encoding='utf-8') as f:
            rows.append(json.load(f))
    return pd.json_normalize(rows)


def assign_phase_regions(df: pd.DataFrame, mse_threshold: float = 0.2, collapse_threshold: float = 0.2) -> pd.Series:
    region = []
    for _, row in df.iterrows():
        mse = row.get('quality.mse', 1.0)
        rare = row.get('tail.rare_failure_rate', 1.0)
        if mse < mse_threshold and rare < collapse_threshold:
            region.append('trainable')
        elif mse < mse_threshold * 1.5:
            region.append('rescue')
        else:
            region.append('failure')
    return pd.Series(region, index=df.index)


def _save_atlas_table(df: pd.DataFrame, output_dir: Path) -> Path:
    parquet_path = output_dir / 'atlas.parquet'
    csv_path = output_dir / 'atlas.csv'
    pkl_path = output_dir / 'atlas.pkl'
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as exc:
        parquet_path.write_text(
            'parquet export unavailable; use atlas.pkl or atlas.csv for full table data. '
            f'error={type(exc).__name__}\n',
            encoding='utf-8',
        )
        return pkl_path


def _load_atlas_table(atlas_path: str | Path) -> pd.DataFrame:
    atlas_path = Path(atlas_path)
    if atlas_path.suffix == '.parquet':
        return pd.read_parquet(atlas_path)
    if atlas_path.suffix == '.pkl':
        return pd.read_pickle(atlas_path)
    if atlas_path.suffix == '.csv':
        return pd.read_csv(atlas_path)
    raise ValueError(f'Unsupported atlas file format: {atlas_path}')


def build_atlas(sweep_dir: str | Path, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_summaries(sweep_dir)
    df['phase_region'] = assign_phase_regions(df)
    return _save_atlas_table(df, output_dir)


def _diag_columns(df: pd.DataFrame) -> list[str]:
    diag_cols = [
        'pathology.support_pix', 'pathology.support_perc', 'pathology.support_ssl',
        'pathology.support_deviation', 'pathology.rho_nor', 'pathology.normal_burden',
        'pathology.conditioning', 'pathology.covariance_conditioning',
        'pathology.relative_shift', 'pathology.prediction_sensitivity',
        'pathology.pathology_score', 'grad_var'
    ]
    return [c for c in diag_cols if c in df.columns]


def _fit_label_pipelines(X_label: pd.DataFrame, y_cls: pd.Series, y_reg: pd.Series):
    cat = ColumnTransformer([('fam', OneHotEncoder(handle_unknown='ignore'), ['target_family'])])
    clf = Pipeline([('cat', cat), ('rf', RandomForestClassifier(n_estimators=200, random_state=0))])
    reg = Pipeline([('cat', cat), ('rf', RandomForestRegressor(n_estimators=200, random_state=0))])
    clf.fit(X_label, y_cls)
    reg.fit(X_label, y_reg)
    return clf, reg


def _fit_combined_pipeline(X_label: pd.DataFrame, X_diag: pd.DataFrame, y_cls: pd.Series, y_reg: pd.Series, diag_cols: list[str]):
    X_comb = pd.concat([X_label, X_diag], axis=1)
    comb_pre = ColumnTransformer([
        ('fam', OneHotEncoder(handle_unknown='ignore'), ['target_family']),
        ('diag', 'passthrough', diag_cols),
    ])
    clf = Pipeline([('pre', comb_pre), ('rf', RandomForestClassifier(n_estimators=200, random_state=0))])
    reg = Pipeline([('pre', comb_pre), ('rf', RandomForestRegressor(n_estimators=200, random_state=0))])
    clf.fit(X_comb, y_cls)
    reg.fit(X_comb, y_reg)
    return clf, reg


def _cross_validated_metrics_label(X_label: pd.DataFrame, y_cls: pd.Series, y_reg: pd.Series) -> dict[str, float]:
    n_splits = max(2, min(5, len(X_label)))
    class_counts = pd.Series(y_cls).value_counts()
    can_stratify = len(np.unique(y_cls)) > 1 and int(class_counts.min()) >= 2 and n_splits <= int(class_counts.min())
    if can_stratify:
        splitter_cls = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        y_true_cls, y_pred_cls, fail_scores = [], [], []
        for tr, te in splitter_cls.split(X_label, y_cls):
            clf, _ = _fit_label_pipelines(X_label.iloc[tr], y_cls.iloc[tr], y_reg.iloc[tr])
            pred = clf.predict(X_label.iloc[te])
            y_true_cls.extend(y_cls.iloc[te].tolist())
            y_pred_cls.extend(pred.tolist())
            if 'failure' in clf.named_steps['rf'].classes_:
                proba = clf.predict_proba(X_label.iloc[te])
                fail_idx = list(clf.named_steps['rf'].classes_).index('failure')
                fail_scores.extend(proba[:, fail_idx].tolist())
            else:
                fail_scores.extend([0.0] * len(te))
        bal_acc = balanced_accuracy_score(y_true_cls, y_pred_cls)
        macro_f1 = f1_score(y_true_cls, y_pred_cls, average='macro')
        y_bin = np.array([1 if y == 'failure' else 0 for y in y_true_cls])
        fail_scores = np.array(fail_scores)
        if y_bin.max() > y_bin.min():
            fail_auroc = roc_auc_score(y_bin, fail_scores)
            pr, re, _ = precision_recall_curve(y_bin, fail_scores)
            fail_auprc = auc(re, pr)
        else:
            fail_auroc, fail_auprc = float('nan'), float('nan')
    else:
        clf, _ = _fit_label_pipelines(X_label, y_cls, y_reg)
        pred = clf.predict(X_label)
        bal_acc = balanced_accuracy_score(y_cls, pred)
        macro_f1 = f1_score(y_cls, pred, average='macro')
        fail_auroc, fail_auprc = float('nan'), float('nan')

    splitter_reg = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    y_true_reg, y_pred_reg = [], []
    for tr, te in splitter_reg.split(X_label):
        _, reg = _fit_label_pipelines(X_label.iloc[tr], y_cls.iloc[tr], y_reg.iloc[tr])
        pred = reg.predict(X_label.iloc[te])
        y_true_reg.extend(y_reg.iloc[te].tolist())
        y_pred_reg.extend(pred.tolist())
    r2 = r2_score(y_true_reg, y_pred_reg)
    return {
        'phase_bal_acc': float(bal_acc),
        'phase_macro_f1': float(macro_f1),
        'failure_auroc': float(fail_auroc),
        'failure_auprc': float(fail_auprc),
        'mse_r2': float(r2),
    }


def _cross_validated_metrics_diag(X_diag: pd.DataFrame, y_cls: pd.Series, y_reg: pd.Series, diag_cols: list[str]) -> dict[str, float]:
    n_splits = max(2, min(5, len(X_diag)))
    class_counts = pd.Series(y_cls).value_counts()
    can_stratify = len(np.unique(y_cls)) > 1 and int(class_counts.min()) >= 2 and n_splits <= int(class_counts.min())
    if can_stratify:
        splitter_cls = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        y_true_cls, y_pred_cls, fail_scores = [], [], []
        for tr, te in splitter_cls.split(X_diag, y_cls):
            clf = RandomForestClassifier(n_estimators=200, random_state=0)
            clf.fit(X_diag.iloc[tr], y_cls.iloc[tr])
            pred = clf.predict(X_diag.iloc[te])
            y_true_cls.extend(y_cls.iloc[te].tolist())
            y_pred_cls.extend(pred.tolist())
            if 'failure' in clf.classes_:
                proba = clf.predict_proba(X_diag.iloc[te])
                fail_idx = list(clf.classes_).index('failure')
                fail_scores.extend(proba[:, fail_idx].tolist())
            else:
                fail_scores.extend([0.0] * len(te))
        bal_acc = balanced_accuracy_score(y_true_cls, y_pred_cls)
        macro_f1 = f1_score(y_true_cls, y_pred_cls, average='macro')
        y_bin = np.array([1 if y == 'failure' else 0 for y in y_true_cls])
        fail_scores = np.array(fail_scores)
        if y_bin.max() > y_bin.min():
            fail_auroc = roc_auc_score(y_bin, fail_scores)
            pr, re, _ = precision_recall_curve(y_bin, fail_scores)
            fail_auprc = auc(re, pr)
        else:
            fail_auroc, fail_auprc = float('nan'), float('nan')
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=0)
        clf.fit(X_diag, y_cls)
        pred = clf.predict(X_diag)
        bal_acc = balanced_accuracy_score(y_cls, pred)
        macro_f1 = f1_score(y_cls, pred, average='macro')
        fail_auroc, fail_auprc = float('nan'), float('nan')

    splitter_reg = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    y_true_reg, y_pred_reg = [], []
    for tr, te in splitter_reg.split(X_diag):
        reg = RandomForestRegressor(n_estimators=200, random_state=0)
        reg.fit(X_diag.iloc[tr], y_reg.iloc[tr])
        pred = reg.predict(X_diag.iloc[te])
        y_true_reg.extend(y_reg.iloc[te].tolist())
        y_pred_reg.extend(pred.tolist())
    r2 = r2_score(y_true_reg, y_pred_reg)
    return {
        'phase_bal_acc': float(bal_acc),
        'phase_macro_f1': float(macro_f1),
        'failure_auroc': float(fail_auroc),
        'failure_auprc': float(fail_auprc),
        'mse_r2': float(r2),
        'diag_cols': diag_cols,
    }


def fit_surrogates(atlas_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_atlas_table(atlas_path)
    df['target_family'] = df['prediction_spec.family'].fillna(df['loss_spec.family'])
    X_label = df[['target_family']].copy()
    y_cls = df['phase_region']
    y_reg = pd.to_numeric(df['quality.mse'], errors='coerce').fillna(df['quality.mse'].mean())

    diag_cols = _diag_columns(df)
    X_diag = df[diag_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    clf_label, reg_label = _fit_label_pipelines(X_label, y_cls, y_reg)
    clf_diag = RandomForestClassifier(n_estimators=200, random_state=0)
    reg_diag = RandomForestRegressor(n_estimators=200, random_state=0)
    clf_diag.fit(X_diag, y_cls)
    reg_diag.fit(X_diag, y_reg)
    clf_comb, reg_comb = _fit_combined_pipeline(X_label, X_diag, y_cls, y_reg, diag_cols)

    metrics = {
        'train_fit': {
            'label_only_acc': float(accuracy_score(y_cls, clf_label.predict(X_label))),
            'diag_only_acc': float(accuracy_score(y_cls, clf_diag.predict(X_diag))),
            'combined_acc': float(accuracy_score(y_cls, clf_comb.predict(pd.concat([X_label, X_diag], axis=1)))),
            'label_only_r2': float(r2_score(y_reg, reg_label.predict(X_label))),
            'diag_only_r2': float(r2_score(y_reg, reg_diag.predict(X_diag))),
            'combined_r2': float(r2_score(y_reg, reg_comb.predict(pd.concat([X_label, X_diag], axis=1)))),
        },
        'cross_validated': {
            'label_only': _cross_validated_metrics_label(X_label, y_cls, y_reg),
            'diagnostics_only': _cross_validated_metrics_diag(X_diag, y_cls, y_reg, diag_cols),
            'combined': _cross_validated_metrics_diag(
                pd.get_dummies(pd.concat([X_label, X_diag], axis=1), columns=['target_family']),
                y_cls,
                y_reg,
                list(pd.get_dummies(pd.concat([X_label, X_diag], axis=1), columns=['target_family']).columns),
            ),
        },
        'diag_cols': diag_cols,
    }
    artifact = {
        'clf_diag': clf_diag,
        'reg_diag': reg_diag,
        'clf_label': clf_label,
        'reg_label': reg_label,
        'clf_comb': clf_comb,
        'reg_comb': reg_comb,
        'diag_cols': diag_cols,
    }
    joblib.dump(artifact, output_dir / 'surrogate.joblib')
    pd.Series(metrics['train_fit']).to_json(output_dir / 'surrogate_metrics_train.json', indent=2)
    with (output_dir / 'surrogate_metrics_cv.json').open('w', encoding='utf-8') as f:
        json.dump(metrics['cross_validated'], f, indent=2)
    with (output_dir / 'surrogate_details.json').open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    return metrics
