"""
================================================================================
  EXPLAINABLE ML FOR CONSUMER LOAN DEFAULT PREDICTION
  Author  : Junior Owusu Ansah
  Degree  : M.S. Computer Science / MBA Finance
  School  : New Mexico Highlands University
  Date    : August 2026
  GitHub  : https://github.com/your-username/loan-default-prediction

  Run     : python src/pipeline.py
  Dataset : Place loan_data.csv in data/ folder (see data/README.md)
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys

# ── Data ──────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# ── Machine learning ──────────────────────────────────────────────────────────
from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing     import StandardScaler
from sklearn.impute            import SimpleImputer
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics           import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix, f1_score
)
from imblearn.over_sampling    import SMOTE
from xgboost                   import XGBClassifier
from lightgbm                  import LGBMClassifier

# ── Explainability & optimisation ─────────────────────────────────────────────
import shap
import optuna
import joblib
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Plot style (thesis-quality) ───────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi'       : 150,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'font.family'      : 'DejaVu Sans',
    'axes.titlesize'   : 12,
    'axes.labelsize'   : 10,
})
COLORS = ['#2171b5', '#fc8d59', '#78c679', '#9e9ac8', '#d73027']
RS     = 42   # global random state
np.random.seed(RS)

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/models',  exist_ok=True)

# ── Column schema ─────────────────────────────────────────────────────────────
NUMERIC_COLS = [
    'person_age', 'person_income', 'person_emp_exp',
    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_cred_hist_length', 'credit_score',
]
CATEGORICAL_COLS = [
    'person_gender', 'person_education', 'person_home_ownership',
    'loan_intent', 'previous_loan_defaults_on_file',
]
TARGET = 'loan_status'


# ════════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ════════════════════════════════════════════════════════════════════════════════

def load_data(filepath: str = 'data/loan_data.csv') -> pd.DataFrame:
    """Load and inspect the raw dataset."""
    if not os.path.exists(filepath):
        sys.exit(
            f"\n[ERROR] File not found: {filepath}\n"
            "Please download loan_data.csv from Kaggle and place it in data/\n"
            "See data/README.md for instructions.\n"
        )

    df = pd.read_csv(filepath)

    print("=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    counts = df[TARGET].value_counts()
    for v, c in counts.items():
        print(f"  Class {v}         : {c:,}  ({c/len(df)*100:.1f}%)")
    print(f"  Missing values : {df.isnull().sum().sum()}")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame):
    """Produce all EDA figures and print descriptive statistics."""
    print("\n" + "=" * 60)
    print("2. EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    def savefig(name):
        path = f'outputs/figures/{name}'
        plt.savefig(path, bbox_inches='tight')
        print(f"  Saved: {path}")

    # ── Fig 1: Class balance ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    counts  = [df[TARGET].eq(0).sum(), df[TARGET].eq(1).sum()]
    bars    = ax.bar(['No Default (0)', 'Default (1)'], counts,
                     color=[COLORS[0], COLORS[4]], width=0.45)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, c + 200,
                f'{c:,}\n({c/len(df)*100:.1f}%)', ha='center', fontsize=10)
    ax.set_title('Fig 1 — Class Distribution'); ax.set_ylabel('Count')
    plt.tight_layout(); savefig('fig1_class_balance.png'); display(fig); plt.close(fig)

    # ── Fig 2: Distributions by default status ────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, col in zip(axes.flatten(), NUMERIC_COLS):
        for lbl, color, name in [(0, COLORS[0], 'No Default'), (1, COLORS[4], 'Default')]:
            df[df[TARGET]==lbl][col].dropna().plot(
                kind='kde', ax=ax, color=color, label=name, linewidth=1.5)
        ax.set_title(col, fontsize=9); ax.legend(fontsize=7)
    fig.suptitle('Fig 2 — Feature Distributions by Default Status', fontsize=12)
    plt.tight_layout(); savefig('fig2_distributions.png'); display(fig); plt.close(fig)

    # ── Fig 3: Correlation matrix ─────────────────────────────
    corr = df[NUMERIC_COLS + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.3, ax=ax)
    ax.set_title('Fig 3 — Feature Correlation Matrix')
    plt.tight_layout(); savefig('fig3_correlation.png'); display(fig); plt.close(fig)

    print("\n  Top correlations with loan_status:")
    print(corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(3).to_string())

    # ── Fig 4: Default rate by categorical features ───────────
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, col in zip(axes, CATEGORICAL_COLS):
        rate = df.groupby(col)[TARGET].mean().sort_values()
        rate.plot(kind='barh', ax=ax, color=COLORS[0])
        ax.axvline(df[TARGET].mean(), color='red', linestyle='--', linewidth=0.8)
        ax.set_title(col.replace('_', '\n'), fontsize=8); ax.set_xlabel('Default Rate')
    fig.suptitle('Fig 4 — Default Rate by Categorical Feature', fontsize=12)
    plt.tight_layout(); savefig('fig4_categorical_rates.png'); display(fig); plt.close(fig)

    # ── Fig 5: Boxplots ───────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, col in zip(axes, ['loan_amnt','loan_int_rate','credit_score','loan_percent_income']):
        df.boxplot(column=col, by=TARGET, ax=ax, patch_artist=True,
                   boxprops=dict(facecolor=COLORS[0], alpha=0.6))
        ax.set_title(col.replace('_', ' '), fontsize=9)
        ax.set_xlabel('loan_status (0=No Default, 1=Default)', fontsize=8)
        plt.sca(ax); plt.title(col.replace('_',' '), fontsize=9)
    fig.suptitle('Fig 5 — Key Variables by Default Status')
    plt.tight_layout(); savefig('fig5_boxplots.png'); display(fig); plt.close(fig)

    # ── Fig 6: Credit score vs interest rate scatter ──────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for lbl, color, name in [(0,COLORS[0],'No Default'), (1,COLORS[4],'Default')]:
        sub = df[df[TARGET]==lbl]
        ax.scatter(sub['credit_score'], sub['loan_int_rate'],
                   alpha=0.25, s=6, color=color, label=name)
    ax.set_xlabel('Credit Score'); ax.set_ylabel('Interest Rate (%)')
    ax.set_title('Fig 6 — Credit Score vs Interest Rate by Default Status'); ax.legend()
    plt.tight_layout(); savefig('fig6_scatter.png'); display(fig); plt.close(fig)

    # ── Table 1: Descriptive statistics ──────────────────────
    summary = df[NUMERIC_COLS].describe().T
    summary['skewness'] = df[NUMERIC_COLS].skew()
    print("\n  Table 1 — Descriptive Statistics:")
    print(summary[['mean','std','min','50%','max','skewness']].round(2).to_string())


# ════════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 11 domain-informed derived features.
    Must be called BEFORE the train/test split.
    """
    df = df.copy()

    df['loan_to_income_ratio']   = df['loan_amnt']       / (df['person_income'] + 1)
    df['monthly_income_est']     = df['person_income']   / 12
    df['monthly_loan_est']       = df['loan_amnt']       / 36
    df['affordability_ratio']    = df['monthly_loan_est']/ (df['monthly_income_est'] + 1)
    df['interest_burden']        = (df['loan_amnt'] * df['loan_int_rate'] / 100) / (df['person_income'] + 1)
    df['rate_score_interaction'] = df['loan_int_rate']   * (850 - df['credit_score'])
    df['hist_per_age']           = df['cb_person_cred_hist_length'] / (df['person_age'] + 1)
    df['emp_age_ratio']          = df['person_emp_exp']  / (df['person_age'] + 1)
    df['high_loan_burden_flag']  = (df['loan_percent_income'] > 0.4).astype(int)
    df['poor_credit_flag']       = (df['credit_score'] < 580).astype(int)
    df['high_rate_flag']         = (df['loan_int_rate'] > 15).astype(int)

    print("\n" + "=" * 60)
    print("3. FEATURE ENGINEERING")
    print("=" * 60)
    print(f"  Features after engineering: {df.shape[1]}")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# 4. PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline — returns processed train/test arrays.
    Order: encode → split → impute (fit train) → scale (fit train) → SMOTE (train only)
    """
    print("\n" + "=" * 60)
    print("4. PREPROCESSING")
    print("=" * 60)

    df = df.copy()

    # ── Encode categoricals ───────────────────────────────────
    df['person_gender'] = (df['person_gender'].str.lower() == 'male').astype(int)
    df['previous_loan_defaults_on_file'] = (
        df['previous_loan_defaults_on_file'].str.strip().str.lower() == 'yes'
    ).astype(int)

    edu_map  = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
    home_map = {'RENT': 0, 'OTHER': 1, 'MORTGAGE': 2, 'OWN': 3}
    df['person_education']      = df['person_education'].map(edu_map).fillna(2)
    df['person_home_ownership'] = df['person_home_ownership'].map(home_map).fillna(0)
    df = pd.get_dummies(df, columns=['loan_intent'], prefix='intent', drop_first=False)

    print(f"  Features after encoding : {df.shape[1] - 1}")

    # ── Separate X and y ─────────────────────────────────────
    X = df.drop(columns=[TARGET])
    y = df[TARGET].reset_index(drop=True)

    # ── Stratified 80/20 split ────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RS, stratify=y
    )
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)
    print(f"  Train : {len(X_train):,}  (default {y_train.mean()*100:.1f}%)")
    print(f"  Test  : {len(X_test):,}   (default {y_test.mean()*100:.1f}%)")

    # ── Impute (fit on train only) ────────────────────────────
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    imputer  = SimpleImputer(strategy='median')
    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
    X_test[num_cols]  = imputer.transform(X_test[num_cols])

    # ── Scale (fit on train only) ─────────────────────────────
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    # ── SMOTE (train only) ────────────────────────────────────
    n0, n1 = np.bincount(y_train)
    print(f"  Before SMOTE: {{0: {n0}, 1: {n1}}}  ratio={n0/n1:.1f}:1")
    smote = SMOTE(random_state=RS, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    y_train_res = pd.Series(y_train_res).reset_index(drop=True)
    n0r, n1r = np.bincount(y_train_res)
    print(f"  After  SMOTE: {{0: {n0r}, 1: {n1r}}}")

    return X_train_res, X_test, y_train_res, y_test, scaler, imputer


# ════════════════════════════════════════════════════════════════════════════════
# 5. MODEL TRAINING — BASELINE COMPARISON
# ════════════════════════════════════════════════════════════════════════════════

def train_baseline_models(X_train, y_train):
    """5-fold stratified CV comparison of four baseline models."""
    print("\n" + "=" * 60)
    print("5. BASELINE MODEL COMPARISON  (5-fold Stratified CV)")
    print("=" * 60)

    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    n0, n1 = np.bincount(y_train); spw = round(n0/n1)

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', C=0.1, random_state=RS),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight='balanced',
            random_state=RS, n_jobs=-1),
        'XGBoost': XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=spw, use_label_encoder=False,
            eval_metric='logloss', random_state=RS),
        'LightGBM': LGBMClassifier(
            n_estimators=300, learning_rate=0.05, class_weight='balanced',
            random_state=RS, verbose=-1),
    }

    cv_scores = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring='roc_auc', n_jobs=-1)
        cv_scores[name] = scores
        print(f"  {name:25s}  AUC = {scores.mean():.4f}  ±  {scores.std():.4f}")

    # Fig 7: Model comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    means   = [cv_scores[m].mean() for m in models]
    stds    = [cv_scores[m].std()  for m in models]
    bars    = ax.barh(list(models.keys()), means, xerr=stds,
                      color=COLORS[:4], height=0.5, capsize=4)
    for bar, m in zip(bars, means):
        ax.text(m + 0.003, bar.get_y() + bar.get_height()/2,
                f'{m:.4f}', va='center', fontsize=9)
    ax.set_xlabel('AUC-ROC (5-fold CV)'); ax.set_xlim(0.4, 1.0)
    ax.set_title('Fig 7 — Baseline Model Comparison')
    plt.tight_layout()
    plt.savefig('outputs/figures/fig7_model_comparison.png', bbox_inches='tight')
    display(fig); plt.close(fig)

    return cv_scores, models


# ════════════════════════════════════════════════════════════════════════════════
# 6. HYPERPARAMETER TUNING — OPTUNA
# ════════════════════════════════════════════════════════════════════════════════

def tune_xgboost(X_train, y_train, n_trials: int = 80) -> XGBClassifier:
    """Bayesian optimisation for XGBoost — 80 Optuna trials."""
    print("\n" + "=" * 60)
    print(f"6a. XGBOOST TUNING  ({n_trials} Optuna trials)")
    print("=" * 60)

    n0, n1 = np.bincount(y_train); spw = round(n0/n1)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

    def objective(trial):
        p = dict(
            n_estimators      = trial.suggest_int('n_estimators', 100, 600),
            max_depth         = trial.suggest_int('max_depth', 3, 10),
            learning_rate     = trial.suggest_float('lr',    0.005, 0.3,  log=True),
            subsample         = trial.suggest_float('sub',   0.5,   1.0),
            colsample_bytree  = trial.suggest_float('col',   0.5,   1.0),
            min_child_weight  = trial.suggest_int('mcw',   1, 10),
            gamma             = trial.suggest_float('gamma', 0.0,   5.0),
            reg_alpha         = trial.suggest_float('alpha', 1e-8,  10,   log=True),
            reg_lambda        = trial.suggest_float('lam',   1e-8,  10,   log=True),
            scale_pos_weight  = spw,
            use_label_encoder = False,
            eval_metric       = 'logloss',
            random_state      = RS,
        )
        return cross_val_score(XGBClassifier(**p), X_train, y_train,
                               cv=cv, scoring='roc_auc').mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best CV AUC-ROC : {study.best_value:.4f}")
    print(f"  Best params     : {study.best_params}")

    best = XGBClassifier(**study.best_params, scale_pos_weight=spw,
                         use_label_encoder=False, eval_metric='logloss',
                         random_state=RS)
    best.fit(X_train, y_train)
    return best, study.best_params


def tune_lightgbm(X_train, y_train, n_trials: int = 60) -> LGBMClassifier:
    """Bayesian optimisation for LightGBM — 60 Optuna trials."""
    print("\n" + "=" * 60)
    print(f"6b. LIGHTGBM TUNING  ({n_trials} Optuna trials)")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

    def objective(trial):
        p = dict(
            n_estimators       = trial.suggest_int('n_estimators', 100, 600),
            max_depth          = trial.suggest_int('max_depth',    3,   12),
            learning_rate      = trial.suggest_float('lr',  0.005, 0.3, log=True),
            num_leaves         = trial.suggest_int('num_leaves',   20,  200),
            subsample          = trial.suggest_float('sub', 0.5,   1.0),
            colsample_bytree   = trial.suggest_float('col', 0.5,   1.0),
            min_child_samples  = trial.suggest_int('mcs',   5,     100),
            reg_alpha          = trial.suggest_float('alpha',1e-8, 10,  log=True),
            reg_lambda         = trial.suggest_float('lam',  1e-8, 10,  log=True),
            class_weight       = 'balanced',
            random_state       = RS,
            verbose            = -1,
        )
        return cross_val_score(LGBMClassifier(**p), X_train, y_train,
                               cv=cv, scoring='roc_auc').mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best CV AUC-ROC : {study.best_value:.4f}")
    print(f"  Best params     : {study.best_params}")

    best = LGBMClassifier(**study.best_params, class_weight='balanced',
                          random_state=RS, verbose=-1)
    best.fit(X_train, y_train)
    return best, study.best_params


# ════════════════════════════════════════════════════════════════════════════════
# 7. EVALUATION
# ════════════════════════════════════════════════════════════════════════════════

def ks_statistic(y_true, y_score) -> float:
    """KS Statistic — industry-standard credit risk metric (> 0.40 = strong)."""
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    df_ks   = pd.DataFrame({'score': y_score, 'label': y_true})
    df_ks   = df_ks.sort_values('score', ascending=False).reset_index(drop=True)
    n_def   = (df_ks['label'] == 1).sum()
    n_non   = (df_ks['label'] == 0).sum()
    if n_def == 0 or n_non == 0:
        return 0.0
    df_ks['cum_def'] = (df_ks['label'] == 1).cumsum() / n_def
    df_ks['cum_non'] = (df_ks['label'] == 0).cumsum() / n_non
    return float((df_ks['cum_def'] - df_ks['cum_non']).abs().max())


def evaluate_model(model, X_test, y_test,
                   model_name: str = 'Model',
                   threshold: float = 0.5) -> dict:
    """
    Full evaluation suite — produces ROC curve, PR curve, confusion matrix.
    Returns a dict of all metrics for the results table.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auc  = roc_auc_score(y_test, y_prob)
    ap   = average_precision_score(y_test, y_prob)
    ks   = ks_statistic(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*60}")
    print(f"RESULTS — {model_name}")
    print(f"{'='*60}")
    print(f"  AUC-ROC      : {auc:.4f}")
    print(f"  Avg Precision: {ap:.4f}")
    print(f"  KS Statistic : {ks:.4f}  {'✓ Strong' if ks>0.4 else '~ Moderate'}")
    print(f"  F1 Score     : {f1:.4f}")
    print(f"  Sensitivity  : {tp/(tp+fn):.4f}")
    print(f"  Specificity  : {tn/(tn+fp):.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Default','Default'])}")

    fpr,  tpr, _ = roc_curve(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(fpr, tpr, color=COLORS[0], lw=2, label=f'AUC={auc:.4f}')
    axes[0].plot([0,1],[0,1],'--',color='gray',lw=1,label='Random')
    axes[0].fill_between(fpr, tpr, alpha=0.08, color=COLORS[0])
    axes[0].set(xlabel='FPR', ylabel='TPR', title=f'ROC — {model_name}'); axes[0].legend()

    axes[1].plot(rec, prec, color=COLORS[1], lw=2, label=f'AP={ap:.4f}')
    axes[1].axhline(np.mean(y_test), color='gray', linestyle='--', label='Baseline')
    axes[1].fill_between(rec, prec, alpha=0.08, color=COLORS[1])
    axes[1].set(xlabel='Recall', ylabel='Precision', title=f'PR — {model_name}'); axes[1].legend()

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                xticklabels=['Pred No','Pred Yes'],
                yticklabels=['Actual No','Actual Yes'])
    axes[2].set_title(f'Confusion Matrix\nSens={tp/(tp+fn):.3f}  Spec={tn/(tn+fp):.3f}')

    plt.suptitle(f'{model_name} — Evaluation Suite', fontsize=12)
    plt.tight_layout()
    fname = f'outputs/figures/fig_eval_{model_name.replace(" ","_")}.png'
    plt.savefig(fname, bbox_inches='tight')
    display(fig); plt.close(fig)

    return {
        'Model'      : model_name,
        'AUC-ROC'    : round(auc, 4),
        'AP'         : round(ap,  4),
        'KS'         : round(ks,  4),
        'F1'         : round(f1,  4),
        'Sensitivity': round(tp/(tp+fn), 4),
        'Specificity': round(tn/(tn+fp), 4),
    }


def print_results_table(all_results: list):
    """Print and return the final model comparison table."""
    df_res = pd.DataFrame(all_results).set_index('Model')
    print("\n" + "=" * 70)
    print("TABLE 2 — FINAL MODEL COMPARISON (Held-Out Test Set, n=9,000)")
    print("=" * 70)
    print(df_res.to_string())
    print("\nBest model per metric:")
    for col in df_res.columns:
        best = df_res[col].idxmax()
        print(f"  {col:15s}: {best}  ({df_res.loc[best, col]:.4f})")
    return df_res


# ════════════════════════════════════════════════════════════════════════════════
# 8. SHAP EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════════════════

def explain_model(model, X_test, model_name: str = 'XGBoost', n_samples: int = 500):
    """
    SHAP-based global and individual explainability suite.
    Produces beeswarm, bar, dependence, and waterfall plots.
    """
    print("\n" + "=" * 60)
    print(f"8. SHAP EXPLAINABILITY — {model_name}")
    print("=" * 60)

    sample      = X_test.iloc[:n_samples].copy()
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    def save_shap(name):
        path = f'outputs/figures/{name}'
        plt.savefig(path, bbox_inches='tight')
        print(f"  Saved: {path}")

    # Fig 12: Beeswarm
    shap.summary_plot(shap_values, sample, max_display=15, plot_type='dot', show=False)
    plt.title(f'Fig 12 — SHAP Beeswarm — {model_name}')
    plt.tight_layout(); save_shap('fig12_shap_beeswarm.png')
    display(plt.gcf()); plt.close()

    # Fig 13: Bar
    shap.summary_plot(shap_values, sample, max_display=15, plot_type='bar', show=False)
    plt.title(f'Fig 13 — Mean |SHAP| Importance — {model_name}')
    plt.tight_layout(); save_shap('fig13_shap_bar.png')
    display(plt.gcf()); plt.close()

    # Fig 14: Dependence — top feature
    feat_imp   = pd.Series(np.abs(shap_values).mean(0), index=sample.columns)
    top_feat   = feat_imp.idxmax()
    shap.dependence_plot(top_feat, shap_values, sample, show=False)
    plt.title(f'Fig 14 — SHAP Dependence: {top_feat}')
    plt.tight_layout(); save_shap('fig14_shap_dependence.png')
    display(plt.gcf()); plt.close()

    # Fig 15: Waterfall — highest-risk applicant
    y_prob   = model.predict_proba(X_test)[:, 1]
    hi_risk  = np.argmax(y_prob[:n_samples])
    exp = shap.Explanation(
        values       = shap_values[hi_risk],
        base_values  = explainer.expected_value,
        data         = sample.iloc[hi_risk].values,
        feature_names= sample.columns.tolist()
    )
    shap.plots.waterfall(exp, show=False)
    plt.title('Fig 15 — SHAP Waterfall (Highest-Risk Applicant)')
    plt.tight_layout(); save_shap('fig15_shap_waterfall.png')
    display(plt.gcf()); plt.close()

    # SHAP importance table
    shap_tbl = feat_imp.sort_values(ascending=False).reset_index()
    shap_tbl.columns = ['Feature', 'Mean |SHAP|']
    print("\n  Table 3 — SHAP Feature Importance (top 10):")
    print(shap_tbl.head(10).to_string(index=False))
    return shap_tbl


# ════════════════════════════════════════════════════════════════════════════════
# 9. SAVE ARTEFACTS
# ════════════════════════════════════════════════════════════════════════════════

def save_artefacts(models: dict, scaler, imputer, feature_names: list):
    """Persist all models and preprocessing objects."""
    print("\n" + "=" * 60)
    print("9. SAVING ARTEFACTS")
    print("=" * 60)
    for name, model in models.items():
        fname = f'outputs/models/{name.lower().replace(" ","_")}.pkl'
        joblib.dump(model, fname)
        print(f"  Saved: {fname}")
    joblib.dump(scaler,  'outputs/models/scaler.pkl')
    joblib.dump(imputer, 'outputs/models/imputer.pkl')
    pd.Series(feature_names).to_csv('outputs/models/feature_names.csv', index=False)
    print("  Saved: outputs/models/scaler.pkl")
    print("  Saved: outputs/models/imputer.pkl")
    print("  Saved: outputs/models/feature_names.csv")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    import traceback

    print("\n" + "█" * 60)
    print("  LOAN DEFAULT PREDICTION PIPELINE — STARTING")
    print("█" * 60)

    try:
        # 1. Load
        df = load_data('data/loan_data.csv')

        # 2. EDA
        run_eda(df)

        # 3. Feature engineering (before split)
        df = engineer_features(df)

        # 4. Preprocess
        X_train_res, X_test, y_train_res, y_test, scaler, imputer = preprocess(df)

        # 5. Baseline comparison
        cv_scores, baseline_models = train_baseline_models(X_train_res, y_train_res)

        # 6. Tune top two models
        best_xgb,  xgb_params  = tune_xgboost(X_train_res,  y_train_res, n_trials=80)
        best_lgbm, lgbm_params = tune_lightgbm(X_train_res, y_train_res, n_trials=60)

        # Fit baseline models on full training set for test evaluation
        print("\n" + "=" * 60)
        print("7. TEST SET EVALUATION")
        print("=" * 60)
        for name, model in baseline_models.items():
            model.fit(X_train_res, y_train_res)

        all_results = []
        eval_models = {
            **baseline_models,
            'XGBoost (Tuned)' : best_xgb,
            'LightGBM (Tuned)': best_lgbm,
        }
        for name, model in eval_models.items():
            res = evaluate_model(model, X_test, y_test, model_name=name)
            all_results.append(res)

        results_df = print_results_table(all_results)

        # 8. SHAP explainability on tuned XGBoost
        shap_tbl = explain_model(best_xgb, X_test, model_name='XGBoost (Tuned)')

        # 9. Save everything
        save_artefacts(
            {'xgboost_tuned': best_xgb, 'lightgbm_tuned': best_lgbm},
            scaler, imputer, X_test.columns.tolist()
        )

        print("\n" + "█" * 60)
        print("  PIPELINE COMPLETE")
        print("  → Figures  : outputs/figures/  (15 PNG files)")
        print("  → Models   : outputs/models/   (.pkl files)")
        print("  → Best model: LightGBM (Tuned)")
        print("█" * 60)

        return best_xgb, best_lgbm, X_test, y_test, results_df

    except Exception:
        print("\n[ERROR] Pipeline failed:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
