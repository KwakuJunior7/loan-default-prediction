"""
================================================================================
  EXPLAINABLE ML FOR CONSUMER LOAN DEFAULT PREDICTION (SYNTHETIC DATA)
  Author  : Junior Owusu Ansah
  School  : New Mexico Highlands University
  GitHub  : https://github.com/KwakuJunior7/loan-default-prediction

  Run     : python src/pipeline.py              (full run, ~30-60 min)
            python src/pipeline.py --trials 10  (quick run)
  Dataset : place loan_data.csv in data/ (see data/README.md)

  Design rules
  - Every preprocessing step that learns from data (imputation, scaling, SMOTE)
    lives inside an imblearn Pipeline, so it is refitted inside each CV fold and
    never sees validation or test rows.
  - person_gender is NOT a model input (ECOA / Regulation B). It is kept aside
    and used only for the fairness check in section 9.
  - previous_loan_defaults_on_file is a deterministic rule in this synthetic
    dataset (every applicant with a prior default has loan_status = 0), so every
    headline model is also reported without it (section 7).
================================================================================
"""

import argparse
import json
import os
import re
import sys
import warnings

warnings.filterwarnings('ignore')

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.rcParams.update({'figure.dpi': 150, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 12, 'axes.labelsize': 10})
COLORS = ['#2171b5', '#fc8d59', '#78c679', '#9e9ac8', '#d73027']
RS = 42
np.random.seed(RS)

FIG = 'outputs/figures'
TAB = 'outputs/tables'
MOD = 'outputs/models'
for d in (FIG, TAB, MOD):
    os.makedirs(d, exist_ok=True)

NUMERIC_COLS = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate',
                'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
CATEGORICAL_COLS = ['person_gender', 'person_education', 'person_home_ownership',
                    'loan_intent', 'previous_loan_defaults_on_file']
TARGET = 'loan_status'                 # 1 = default, 0 = no default (see README, "About the label")
PROTECTED = 'person_gender'            # never a model input
ARTIFACT = 'previous_loan_defaults_on_file'


def savefig(name):
    path = f'{FIG}/{name}'
    plt.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f'  Saved: {path}')


def save_table(df, name, index=True):
    df.to_csv(f'{TAB}/{name}.csv', index=index)
    print(f'  Saved: {TAB}/{name}.csv')


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ════════════════════════════════════════════════════════════════════════════

def load_data(filepath='data/loan_data.csv'):
    if not os.path.exists(filepath):
        sys.exit(f'\n[ERROR] File not found: {filepath}\nSee data/README.md for download instructions.\n')
    df = pd.read_csv(filepath)
    print('=' * 60 + '\n1. DATASET\n' + '=' * 60)
    print(f'  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns; missing values: {int(df.isnull().sum().sum())}')
    bad_age = df['person_age'] > 100
    print(f'  Dropping {int(bad_age.sum())} rows with person_age > 100 (data-entry artefacts, max was {df.person_age.max():.0f})')
    df = df.loc[~bad_age].reset_index(drop=True)
    for v, c in df[TARGET].value_counts().sort_index().items():
        print(f'  loan_status = {v}: {c:,} ({c / len(df):.1%})')
    return df


def audit_dataset(df):
    """Checks that show this dataset is synthetic and where its rules lie."""
    print('\n' + '=' * 60 + '\n2. DATA AUDIT\n' + '=' * 60)
    ct = pd.crosstab(df[ARTIFACT], df[TARGET], margins=True)
    print('  Prior default on file vs loan_status:\n' + ct.to_string())
    save_table(ct, 'audit_prior_default_vs_target')
    yes = df[df[ARTIFACT] == 'Yes']
    print(f'  -> {len(yes):,} applicants ({len(yes) / len(df):.1%}) have a prior default; '
          f'their default rate is {yes[TARGET].mean():.1%}. This is a hard-coded rule, not behaviour.')
    q = df.groupby(pd.qcut(df['credit_score'], 5), observed=True)[TARGET].mean()
    print('  Default rate by credit-score quintile (flat = no relationship):')
    print('    ' + ', '.join(f'{r:.1%}' for r in q.values))
    means = df.groupby(TARGET)[['loan_int_rate', 'loan_percent_income', 'person_income']].mean()
    print('  Means by loan_status (supports 1 = default):\n' + means.round(3).to_string())
    save_table(means.round(4), 'audit_means_by_target')
    audit = {
        'n_rows': int(len(df)),
        'default_rate': round(float(df[TARGET].mean()), 4),
        'prior_default_share': round(len(yes) / len(df), 4),
        'default_rate_given_prior_default': round(float(yes[TARGET].mean()), 4),
        'default_rate_no_prior_default': round(float(df.loc[df[ARTIFACT] == 'No', TARGET].mean()), 4),
        'default_rate_by_credit_score_quintile': [round(float(x), 4) for x in q.values],
        'default_rate_by_gender': {k: round(float(v), 4) for k, v in df.groupby(PROTECTED)[TARGET].mean().items()},
    }
    return audit


def run_eda(df):
    print('\n' + '=' * 60 + '\n3. EXPLORATORY DATA ANALYSIS\n' + '=' * 60)
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = [df[TARGET].eq(0).sum(), df[TARGET].eq(1).sum()]
    bars = ax.bar(['No default (0)', 'Default (1)'], counts, color=[COLORS[0], COLORS[4]], width=0.45)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 200, f'{c:,}\n({c / len(df):.1%})', ha='center', fontsize=10)
    ax.set_title('Class distribution'); ax.set_ylabel('Applicants')
    plt.tight_layout(); savefig('fig1_class_balance.png')

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, col in zip(axes.flatten(), NUMERIC_COLS):
        for lbl, color, name in [(0, COLORS[0], 'No default'), (1, COLORS[4], 'Default')]:
            df.loc[df[TARGET] == lbl, col].plot(kind='kde', ax=ax, color=color, label=name, linewidth=1.5)
        ax.set_title(col, fontsize=9); ax.legend(fontsize=7)
    fig.suptitle('Feature distributions by default status')
    plt.tight_layout(); savefig('fig2_distributions.png')

    corr = df[NUMERIC_COLS + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.3, ax=ax)
    ax.set_title('Correlation matrix (numeric features)')
    plt.tight_layout(); savefig('fig3_correlation.png')

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, col in zip(axes, CATEGORICAL_COLS):
        df.groupby(col)[TARGET].mean().sort_values().plot(kind='barh', ax=ax, color=COLORS[0])
        ax.axvline(df[TARGET].mean(), color='red', linestyle='--', linewidth=0.8)
        ax.set_title(col.replace('_', ' '), fontsize=9); ax.set_xlabel('Default rate')
    fig.suptitle('Default rate by categorical feature (red line = overall rate)')
    plt.tight_layout(); savefig('fig4_categorical_rates.png')


# ════════════════════════════════════════════════════════════════════════════
# 4. FEATURES
# ════════════════════════════════════════════════════════════════════════════

def engineer_features(df):
    """Row-wise derived features (no statistics are learned, so this is leakage-free)."""
    df = df.copy()
    df['loan_to_income_ratio'] = df['loan_amnt'] / (df['person_income'] + 1)
    df['interest_burden'] = (df['loan_amnt'] * df['loan_int_rate'] / 100) / (df['person_income'] + 1)
    df['rate_score_interaction'] = df['loan_int_rate'] * (850 - df['credit_score'])
    df['hist_per_age'] = df['cb_person_cred_hist_length'] / (df['person_age'] + 1)
    df['emp_age_ratio'] = df['person_emp_exp'] / (df['person_age'] + 1)
    df['high_loan_burden_flag'] = (df['loan_percent_income'] > 0.4).astype(int)
    df['poor_credit_flag'] = (df['credit_score'] < 580).astype(int)
    df['high_rate_flag'] = (df['loan_int_rate'] > 15).astype(int)

    df[ARTIFACT] = (df[ARTIFACT].str.strip().str.lower() == 'yes').astype(int)
    edu_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
    home_map = {'RENT': 0, 'OTHER': 1, 'MORTGAGE': 2, 'OWN': 3}
    df['person_education'] = df['person_education'].map(edu_map)
    df['person_home_ownership'] = df['person_home_ownership'].map(home_map)
    df = pd.get_dummies(df, columns=['loan_intent'], prefix='intent', dtype=int)
    print('\n' + '=' * 60 + '\n4. FEATURES\n' + '=' * 60)
    return df


def split(df):
    y = df[TARGET]
    gender = df[PROTECTED]
    X = df.drop(columns=[TARGET, PROTECTED])
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, gender, test_size=0.20, random_state=RS, stratify=y)
    print(f'  Model features: {X.shape[1]} (person_gender excluded)')
    print(f'  Train: {len(X_tr):,} (default {y_tr.mean():.1%})   Test: {len(X_te):,} (default {y_te.mean():.1%})')
    return X_tr, X_te, y_tr, y_te, g_te


# ════════════════════════════════════════════════════════════════════════════
# 5. MODELS (every learned step inside the pipeline)
# ════════════════════════════════════════════════════════════════════════════

def make_pipe(model, scale=False):
    steps = [('impute', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scale', StandardScaler()))
    steps += [('smote', SMOTE(random_state=RS)), ('model', model)]
    return Pipeline(steps)


def baseline_models():
    return {
        'Logistic Regression': make_pipe(LogisticRegression(max_iter=2000, C=0.1, random_state=RS), scale=True),
        'Random Forest': make_pipe(RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RS, n_jobs=-1)),
        'XGBoost': make_pipe(XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                           eval_metric='logloss', random_state=RS, n_jobs=-1)),
        'LightGBM': make_pipe(LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=RS,
                                             verbose=-1, n_jobs=-1)),
    }


CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)


def compare_baselines(X_tr, y_tr):
    print('\n' + '=' * 60 + '\n5. BASELINE COMPARISON (5-fold CV, SMOTE inside each fold)\n' + '=' * 60)
    rows = []
    for name, pipe in baseline_models().items():
        s = cross_val_score(pipe, X_tr, y_tr, cv=CV, scoring='roc_auc', n_jobs=-1)
        rows.append({'Model': name, 'CV AUC mean': round(s.mean(), 4), 'CV AUC sd': round(s.std(), 4)})
        print(f'  {name:22s} AUC = {s.mean():.4f} +/- {s.std():.4f}')
    res = pd.DataFrame(rows).set_index('Model')
    save_table(res, 'cv_baselines')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(res.index, res['CV AUC mean'], xerr=res['CV AUC sd'], color=COLORS[:4], height=0.5, capsize=4)
    for i, m in enumerate(res['CV AUC mean']):
        ax.text(m + 0.003, i, f'{m:.4f}', va='center', fontsize=9)
    ax.set_xlim(0.8, 1.0); ax.set_xlabel('ROC-AUC (5-fold CV, training set)')
    ax.set_title('Baseline model comparison')
    plt.tight_layout(); savefig('fig5_model_comparison.png')
    return res


def tune(kind, X_tr, y_tr, n_trials):
    print(f'\n  Tuning {kind} ({n_trials} Optuna trials, SMOTE inside each fold)')

    def build(p):
        if kind == 'XGBoost':
            return make_pipe(XGBClassifier(**p, eval_metric='logloss', random_state=RS, n_jobs=-1))
        return make_pipe(LGBMClassifier(**p, random_state=RS, verbose=-1, n_jobs=-1))

    def objective(trial):
        if kind == 'XGBoost':
            p = dict(n_estimators=trial.suggest_int('n_estimators', 100, 600),
                     max_depth=trial.suggest_int('max_depth', 3, 10),
                     learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                     subsample=trial.suggest_float('subsample', 0.5, 1.0),
                     colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                     min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
                     gamma=trial.suggest_float('gamma', 0.0, 5.0),
                     reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
                     reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10, log=True))
        else:
            p = dict(n_estimators=trial.suggest_int('n_estimators', 100, 600),
                     max_depth=trial.suggest_int('max_depth', 3, 12),
                     learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                     num_leaves=trial.suggest_int('num_leaves', 20, 200),
                     subsample=trial.suggest_float('subsample', 0.5, 1.0), subsample_freq=1,
                     colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                     min_child_samples=trial.suggest_int('min_child_samples', 5, 100),
                     reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
                     reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10, log=True))
        return cross_val_score(build(p), X_tr, y_tr, cv=CV, scoring='roc_auc', n_jobs=-1).mean()

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RS))
    study.optimize(objective, n_trials=n_trials)
    params = dict(study.best_params)
    if kind == 'LightGBM':
        params['subsample_freq'] = 1
    print(f'  Best CV AUC: {study.best_value:.4f}')
    return build, params, round(study.best_value, 4)


# ════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def ks_statistic(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(pipe, X_te, y_te, name, threshold=0.5, plot=True):
    p = pipe.predict_proba(X_te)[:, 1]
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    row = {'Model': name, 'AUC-ROC': roc_auc_score(y_te, p), 'Avg precision': average_precision_score(y_te, p),
           'KS': ks_statistic(y_te, p), 'F1': f1_score(y_te, pred), 'Recall (sensitivity)': tp / (tp + fn),
           'Specificity': tn / (tn + fp)}
    row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
    print(f"  {name:38s} AUC {row['AUC-ROC']:.4f}  AP {row['Avg precision']:.4f}  KS {row['KS']:.4f}  F1 {row['F1']:.4f}")
    if plot:
        fpr, tpr, _ = roc_curve(y_te, p)
        prec, rec, _ = precision_recall_curve(y_te, p)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(fpr, tpr, color=COLORS[0], lw=2, label=f"AUC={row['AUC-ROC']:.3f}")
        axes[0].plot([0, 1], [0, 1], '--', color='gray', lw=1)
        axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC'); axes[0].legend()
        axes[1].plot(rec, prec, color=COLORS[1], lw=2, label=f"AP={row['Avg precision']:.3f}")
        axes[1].axhline(y_te.mean(), color='gray', linestyle='--', label='Base rate')
        axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision-recall'); axes[1].legend()
        sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues', ax=axes[2],
                    xticklabels=['Pred no default', 'Pred default'], yticklabels=['No default', 'Default'])
        axes[2].set_title(f"Confusion matrix (threshold {threshold})")
        fig.suptitle(f'{name}: held-out test set')
        slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
        plt.tight_layout(); savefig(f'fig_eval_{slug}.png')
    return row, p


# ════════════════════════════════════════════════════════════════════════════
# 8. SHAP
# ════════════════════════════════════════════════════════════════════════════

def explain(pipe, X_te, tag, title, n_samples=1000):
    print(f'\n  SHAP: {title}')
    sample = X_te.sample(n=min(n_samples, len(X_te)), random_state=RS)
    Xs = pd.DataFrame(pipe.named_steps['impute'].transform(sample), columns=sample.columns, index=sample.index)
    explainer = shap.TreeExplainer(pipe.named_steps['model'])
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[1]
    base = explainer.expected_value
    base = base[1] if np.ndim(base) else base

    shap.summary_plot(sv, Xs, max_display=15, show=False)
    plt.title(f'SHAP summary: {title}'); plt.tight_layout(); savefig(f'fig_shap_beeswarm_{tag}.png')
    imp = pd.Series(np.abs(sv).mean(0), index=Xs.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.head(15)[::-1].plot(kind='barh', ax=ax, color=COLORS[0])
    ax.set_xlabel('Mean |SHAP| (log-odds)'); ax.set_title(f'Feature importance: {title}')
    plt.tight_layout(); savefig(f'fig_shap_bar_{tag}.png')

    p = pipe.predict_proba(sample)[:, 1]
    i = int(np.argmax(p))
    exp = shap.Explanation(values=sv[i], base_values=base, data=Xs.iloc[i].values, feature_names=list(Xs.columns))
    shap.plots.waterfall(exp, max_display=12, show=False)
    plt.title(f'Highest-risk applicant in sample (p = {p[i]:.3f})'); plt.tight_layout()
    savefig(f'fig_shap_waterfall_{tag}.png')

    tbl = imp.round(4).rename('Mean |SHAP|').to_frame()
    tbl['Share of total'] = (imp / imp.sum()).round(4)
    save_table(tbl, f'shap_importance_{tag}')
    return tbl


# ════════════════════════════════════════════════════════════════════════════
# 9. FAIRNESS CHECK (gender is used here only)
# ════════════════════════════════════════════════════════════════════════════

def fairness(p, y_te, g_te, threshold=0.5):
    print('\n' + '=' * 60 + '\n9. FAIRNESS CHECK BY GENDER (gender is not a model input)\n' + '=' * 60)
    df = pd.DataFrame({'p': p, 'y': y_te.values, 'g': g_te.values})
    df['pred'] = (df.p >= threshold).astype(int)
    rows = []
    for g, d in df.groupby('g'):
        rows.append({'Group': g, 'Applicants': len(d), 'Actual default rate': d.y.mean(),
                     'Predicted default rate': d.pred.mean(),
                     'True positive rate': d.loc[d.y == 1, 'pred'].mean(),
                     'False positive rate': d.loc[d.y == 0, 'pred'].mean(),
                     'AUC': roc_auc_score(d.y, d.p)})
    t = pd.DataFrame(rows).set_index('Group').round(4)
    approve = 1 - t['Predicted default rate']
    t['Approval rate'] = approve.round(4)
    t['Adverse impact ratio'] = (approve / approve.max()).round(4)
    print(t.to_string())
    save_table(t, 'fairness_by_gender')
    return t


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=50, help='Optuna trials per tuned model')
    ap.add_argument('--data', default='data/loan_data.csv')
    a = ap.parse_args()

    df = load_data(a.data)
    audit = audit_dataset(df)
    run_eda(df)
    df = engineer_features(df)
    X_tr, X_te, y_tr, y_te, g_te = split(df)

    cv_res = compare_baselines(X_tr, y_tr)

    print('\n' + '=' * 60 + '\n6. HYPERPARAMETER TUNING\n' + '=' * 60)
    tuned = {}
    for kind in ('XGBoost', 'LightGBM'):
        build, params, cv_auc = tune(kind, X_tr, y_tr, a.trials)
        tuned[kind] = (build, params, cv_auc)

    print('\n' + '=' * 60 + '\n7. HELD-OUT TEST SET (n = {:,})\n'.format(len(X_te)) + '=' * 60)
    rows, fitted, probs = [], {}, {}
    for name, pipe in baseline_models().items():
        pipe.fit(X_tr, y_tr); fitted[name] = pipe
        r, probs[name] = evaluate(pipe, X_te, y_te, name); rows.append(r)
    for kind, (build, params, _) in tuned.items():
        name = f'{kind} (tuned)'
        pipe = build(params); pipe.fit(X_tr, y_tr); fitted[name] = pipe
        r, probs[name] = evaluate(pipe, X_te, y_te, name); rows.append(r)
    results = pd.DataFrame(rows).set_index('Model')
    best_name = results['AUC-ROC'].idxmax()
    best_kind = best_name.replace(' (tuned)', '') if '(tuned)' in best_name else 'LightGBM'
    print(f'  Best by AUC-ROC: {best_name}')

    # Sensitivity analysis: the same tuned settings without the synthetic rule feature
    print('\n  Without previous_loan_defaults_on_file (artifact removed):')
    X_tr_b, X_te_b = X_tr.drop(columns=[ARTIFACT]), X_te.drop(columns=[ARTIFACT])
    build, params, _ = tuned[best_kind]
    pipe_b = build(params); pipe_b.fit(X_tr_b, y_tr)
    r_b, _ = evaluate(pipe_b, X_te_b, y_te, f'{best_kind} (tuned), no prior-default feature')
    rows.append(r_b)
    # The real modelling problem: applicants the synthetic rule does not decide
    no_prior = X_te[ARTIFACT] == 0
    r_sub, _ = evaluate(fitted[f'{best_kind} (tuned)'], X_te[no_prior], y_te[no_prior],
                        f'{best_kind} (tuned), applicants without a prior default', plot=False)
    rows.append(r_sub)
    results = pd.DataFrame(rows).set_index('Model')
    save_table(results, 'test_results')

    print('\n' + '=' * 60 + '\n8. SHAP EXPLAINABILITY\n' + '=' * 60)
    shap_all = explain(fitted[f'{best_kind} (tuned)'], X_te, 'all_features', f'{best_kind} (tuned), all features')
    shap_b = explain(pipe_b, X_te_b, 'no_prior_default', f'{best_kind} (tuned), without prior-default feature')

    fair = fairness(probs[f'{best_kind} (tuned)'], y_te, g_te)

    joblib.dump(fitted[f'{best_kind} (tuned)'], f'{MOD}/{best_kind.lower()}_tuned_pipeline.joblib')
    summary = {
        'audit': audit,
        'cv_baselines': cv_res.to_dict(orient='index'),
        'tuned': {k: {'cv_auc': v[2], 'params': v[1]} for k, v in tuned.items()},
        'test_results': results.to_dict(orient='index'),
        'shap_top10_all_features': shap_all.head(10).to_dict(orient='index'),
        'shap_top10_no_prior_default': shap_b.head(10).to_dict(orient='index'),
        'fairness_by_gender': fair.to_dict(orient='index'),
        'n_test': int(len(X_te)), 'optuna_trials': a.trials,
    }
    with open('outputs/results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print('\n  Saved: outputs/results.json\n  PIPELINE COMPLETE')


if __name__ == '__main__':
    main()
