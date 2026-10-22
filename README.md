# Explainable Machine Learning for Consumer Loan Default Prediction
### A Comparative Model Study with SHAP-Based Interpretability

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Framework-scikit--learn%20%7C%20XGBoost%20%7C%20LightGBM-orange" />
  <img src="https://img.shields.io/badge/Explainability-SHAP-purple" />
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen" />
</p>

---

> **Author:** Junior Owusu Ansah  
> **Degree:** M.S. Computer Science / MBA Finance — New Mexico Highlands University  
> **Contact:** [Owususjunior7@gmail.com](mailto:Owususjunior7@gmail.com) | [jowusuansah@live.nmhu.edu](mailto:jowusuansah@live.nmhu.edu)  
> **Date:** August 2026  
> **Dataset:** [Kaggle — Bank Loan Data (45,000 applications)](https://www.kaggle.com/datasets/udaymalviya/bank-loan-data)

---

## Abstract

Machine learning is becoming increasingly critical for lenders in scoring consumer loan applications, but responsible deployment requires understanding not only *how accurate* a model is, but *why* it produces the scores it does. This project constructs and benchmarks four classification models — Logistic Regression, Random Forest, XGBoost, and LightGBM — on a dataset of 45,000 U.S. consumer loan applications with a 22.2% observed default rate. The top two models are subsequently tuned using Optuna Bayesian optimisation.

The best model, **LightGBM (Tuned)**, achieves an **AUC-ROC of 0.9785**, an **Average Precision of 0.9407**, and a **KS Statistic of 0.8291** on the held-out test set — all production-grade results by industry standards. SHAP (SHapley Additive exPlanations) values are then computed on the tuned XGBoost model to produce both global and individual-level explanations. The analysis reveals that a single behavioural feature — `previous_loan_defaults_on_file` — contributes more than **seven times** as much to the average prediction as the next most important variable (`person_income`), and does so in a direction that is counterintuitive to conventional credit-underwriting logic. This finding illustrates both the power and the risk of black-box ensemble methods in regulated lending: aggregate accuracy metrics can mask a model's near-total dependence on one variable whose real-world validity has not been independently verified.

**Keywords:** Credit Risk, Loan Default Prediction, Machine Learning, XGBoost, LightGBM, SHAP, Explainable AI, SMOTE, Optuna, Consumer Finance, Regulatory Compliance

---

## Table of Contents

1. [Project Motivation](#1-project-motivation)
2. [Dataset](#2-dataset)
3. [Repository Structure](#3-repository-structure)
4. [Installation](#4-installation)
5. [Methodology Overview](#5-methodology-overview)
6. [Key Results](#6-key-results)
7. [SHAP Explainability Findings](#7-shap-explainability-findings)
8. [Regulatory & Compliance Implications](#8-regulatory--compliance-implications)
9. [Limitations & Future Work](#9-limitations--future-work)
10. [How to Reproduce](#10-how-to-reproduce)
11. [References](#11-references)
12. [Citation](#12-citation)

---

## 1. Project Motivation

Credit risk assessment is one of the most consequential applications of machine learning in modern finance. In the United States, lenders are required under the **Equal Credit Opportunity Act (ECOA)** and its implementing rule, **Regulation B**, to provide applicants with specific, articulable reasons for adverse credit decisions. This regulatory requirement creates a direct conflict with the "black box" nature of the gradient-boosted tree models that currently dominate tabular financial-data benchmarks.

This project addresses that tension directly. It does not simply maximise a model's AUC-ROC and report the number. Instead, it builds a **complete, auditable, explainable pipeline**: raw data → cleaning → feature engineering → model comparison → Bayesian hyperparameter optimisation → held-out test evaluation → SHAP-based interpretability at both the population and individual-applicant levels.

The project is designed from the perspective of a finance professional who understands both the **business context of credit risk** (probability of default, loss given default, expected loss, IFRS 9 provisioning) and the **technical mechanics** of the models being deployed.

---

## 2. Dataset

| Property | Value |
|---|---|
| **Source** | Kaggle — [Bank Loan Data](https://www.kaggle.com/datasets/udaymalviya/bank-loan-data) |
| **Observations** | 45,000 loan applications |
| **Features** | 14 (8 numeric, 5 categorical, 1 target) |
| **Target** | `loan_status` — 1 = Default, 0 = No Default |
| **Default rate** | 22.2% (10,000 defaults / 35,000 non-defaults) |
| **Missing values** | None |

### Feature Inventory

| Feature | Type | Description |
|---|---|---|
| `person_age` | Numeric | Applicant age (years) |
| `person_gender` | Categorical | Gender (male / female) |
| `person_education` | Categorical | Highest education level |
| `person_income` | Numeric | Annual income (USD) |
| `person_emp_exp` | Numeric | Years of employment experience |
| `person_home_ownership` | Categorical | RENT / OWN / MORTGAGE / OTHER |
| `loan_amnt` | Numeric | Requested loan amount (USD) |
| `loan_intent` | Categorical | Loan purpose (6 categories) |
| `loan_int_rate` | Numeric | Annual interest rate (%) |
| `loan_percent_income` | Numeric | Loan amount as % of annual income |
| `cb_person_cred_hist_length` | Numeric | Credit history length (years) |
| `credit_score` | Numeric | Credit bureau score (390–850) |
| `previous_loan_defaults_on_file` | Categorical | Prior default on record (Yes / No) |
| `loan_status` | **Target** | **1 = Default, 0 = No Default** |

### Descriptive Statistics

| Statistic | person_age | person_income ($) | loan_amnt ($) | loan_int_rate (%) | credit_score |
|---|---|---|---|---|---|
| Mean | 27.76 | 80,319 | 9,583 | 11.01 | 632.6 |
| Std. Dev. | 6.05 | 80,423 | 6,315 | 2.98 | 50.4 |
| Min | 20 | 8,000 | 500 | 5.42 | 390 |
| Median | 26 | 67,048 | 8,000 | 11.01 | 640 |
| Max | 144* | 7,200,766* | 35,000 | 20.00 | 850 |

*\* Outlier values flagged — max age of 144 and max income of $7.2M exceed the 75th percentile by an extreme margin. These were retained but represent a data-quality issue that is addressed in the Limitations section.*

---

## 3. Repository Structure

```
loan-default-prediction/
│
├── data/
│   └── loan_data.csv                      # Raw dataset (download from Kaggle)
│
├── notebooks/
│   └── loan_default_thesis_pipeline.ipynb # Full annotated Jupyter notebook
│
├── src/
│   └── loan_default_pipeline_custom.py    # Production-grade Python pipeline
│
├── outputs/
│   ├── figures/                           # All thesis figures (PNG)
│   │   ├── fig1_class_balance.png
│   │   ├── fig2_distributions.png
│   │   ├── fig3_correlation.png
│   │   ├── fig4_categorical_rates.png
│   │   ├── fig5_boxplots.png
│   │   ├── fig6_scatter.png
│   │   ├── fig7_model_comparison.png
│   │   ├── fig_eval_XGBoost.png
│   │   ├── fig_eval_LightGBM_tuned.png
│   │   ├── fig12_shap_beeswarm.png
│   │   ├── fig13_shap_bar.png
│   │   ├── fig14_shap_dependence.png
│   │   └── fig15_shap_waterfall.png
│   └── models/
│       ├── xgboost_tuned_model.pkl
│       ├── lightgbm_tuned_model.pkl
│       ├── scaler.pkl
│       └── imputer.pkl
│
├── requirements.txt
└── README.md
```

---

## 4. Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/loan-default-prediction.git
cd loan-default-prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Download dataset from Kaggle and place in data/
#    https://www.kaggle.com/datasets/udaymalviya/bank-loan-data
```

### `requirements.txt`

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
imbalanced-learn>=0.11.0
shap>=0.44.0
optuna>=3.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
statsmodels>=0.14.0
joblib>=1.3.0
ipython>=8.0.0
jupyter>=1.0.0
```

---

## 5. Methodology Overview

### 5.1 Pipeline Architecture

```
Raw Data (45,000 rows × 14 cols)
        │
        ▼
Exploratory Data Analysis (EDA)
  ├── Class imbalance check (22.2% default)
  ├── Feature distributions by default status
  ├── Correlation matrix
  └── Categorical default rates
        │
        ▼
Feature Engineering (+11 derived features → 29 total)
  ├── loan_to_income_ratio
  ├── affordability_ratio
  ├── interest_burden
  ├── rate_score_interaction
  ├── emp_age_ratio
  ├── hist_per_age
  ├── high_loan_burden_flag (loan_pct_income > 40%)
  ├── poor_credit_flag (credit_score < 580)
  ├── high_rate_flag (int_rate > 15%)
  └── One-hot encoded loan_intent (6 dummies)
        │
        ▼
Preprocessing
  ├── Stratified 80/20 train/test split
  ├── Median imputation (fit on train only)
  ├── StandardScaler (fit on train only)
  └── SMOTE oversampling (training set only → 28,000 per class)
        │
        ▼
Model Training & Comparison (5-fold Stratified CV)
  ├── Logistic Regression (balanced class weights)
  ├── Random Forest (n=300, balanced)
  ├── XGBoost (baseline)
  └── LightGBM (baseline)
        │
        ▼
Hyperparameter Optimisation (Optuna — Bayesian)
  ├── XGBoost: 80 trials → best CV AUC = 0.9943
  └── LightGBM: 60 trials → best CV AUC = 0.9942
        │
        ▼
Held-Out Test Evaluation (n = 9,000)
  ├── AUC-ROC, Average Precision, KS Statistic
  ├── F1, Sensitivity, Specificity
  └── ROC curves, PR curves, Confusion matrices
        │
        ▼
SHAP Explainability (tuned XGBoost)
  ├── Global: Beeswarm + Mean |SHAP| bar
  ├── Feature-level: Dependence plot
  └── Individual: Waterfall (highest-risk applicant)
```

### 5.2 Encoding Strategy

| Feature | Strategy | Rationale |
|---|---|---|
| `person_gender` | Binary (male=1) | Two categories, no ordinal relationship |
| `person_education` | Ordinal (HS=0 → PhD=4) | Natural educational hierarchy |
| `person_home_ownership` | Ordinal (RENT=0 → OWN=3) | Financial stability proxy |
| `loan_intent` | One-hot (6 dummies) | No natural ordering among purposes |
| `previous_loan_defaults_on_file` | Binary (Yes=1, No=0) | Most predictive feature in the dataset |

### 5.3 Class Imbalance Handling

The training set had a 3.5:1 imbalance ratio (28,000 non-defaults vs. 8,000 defaults). **SMOTE** (Synthetic Minority Over-sampling Technique) was applied exclusively to the training set, generating synthetic minority-class observations to achieve a balanced 28,000:28,000 distribution. The test set was never touched, preserving the natural class distribution for all reported performance metrics. This prevents data leakage and ensures that reported metrics reflect real-world performance (He & Garcia, 2009).

---

## 6. Key Results

### 6.1 Cross-Validation Comparison (Training Set)

| Model | Mean AUC-ROC | Std. Dev. |
|---|---|---|
| Logistic Regression | 0.9693 | ±0.0008 |
| Random Forest | 0.9785 | ±0.0004 |
| XGBoost | 0.9912 | ±0.0004 |
| LightGBM | 0.9928 | ±0.0002 |

All three tree-based ensembles outperform the logistic regression baseline by a statistically meaningful margin. LightGBM achieves the highest cross-validated AUC-ROC, consistent with findings in the broader literature (Zhi et al., 2025; Melsom et al., 2022).

### 6.2 Hyperparameter Optimisation (Optuna)

| Model | Trials | Best CV AUC-ROC | Key Parameters |
|---|---|---|---|
| XGBoost (Tuned) | 80 | **0.9943** | n=568, depth=10, lr=0.0496, subsample=0.779 |
| LightGBM (Tuned) | 60 | **0.9942** | n=514, depth=11, leaves=189, lr=0.0846, subsample=0.638 |

Optuna's Bayesian optimisation was chosen over grid search for its computational efficiency — prior research finds Optuna to be 10.7× faster than grid search for XGBoost and 40.5× faster for LightGBM (Souadda et al., 2025).

### 6.3 Final Test Set Results (n = 9,000 held-out applications)

| Model | AUC-ROC | Avg Precision | KS Stat | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9568 | 0.8696 | 0.7616 | 0.7550 | 0.8775 | 0.8723 |
| Random Forest | 0.9709 | 0.9223 | 0.8034 | 0.8014 | **0.8885** | 0.9060 |
| XGBoost | 0.9743 | 0.9314 | 0.8045 | 0.8328 | 0.8295 | 0.9536 |
| LightGBM | 0.9767 | 0.9373 | 0.8189 | 0.8439 | 0.8215 | 0.9641 |
| XGBoost (Tuned) | 0.9773 | 0.9386 | 0.8254 | 0.8472 | 0.8260 | 0.9646 |
| **LightGBM (Tuned)** | **0.9785** | **0.9407** | **0.8291** | **0.8537** | 0.8270 | **0.9684** |

**Best model: LightGBM (Tuned)** — leads on 5 of 6 metrics. A KS statistic of 0.8291 is exceptional by industry standards, where scores above 0.60 are already considered strong for production consumer credit scorecards (Jakubik & Teleu, 2025). Random Forest achieves the highest sensitivity (0.8885), making it the better choice in contexts where minimising false negatives (missed defaults) is the primary objective.

> **Metric interpretation note:** In a credit risk context, **sensitivity** (true positive rate) measures the fraction of actual defaulters correctly flagged. **Specificity** measures the fraction of non-defaulters correctly approved. **KS statistic** measures the maximum separation between the cumulative default and non-default distributions — the primary discriminatory metric used by credit risk departments in banking.

---

## 7. SHAP Explainability Findings

SHAP values were computed on the tuned XGBoost model across 500 randomly sampled test observations.

### 7.1 Global Feature Importance

The mean absolute SHAP values reveal a striking concentration of predictive power:

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `previous_loan_defaults_on_file` | **5.006** |
| 2 | `person_income` | 0.740 |
| 3 | `loan_to_income_ratio` | 0.681 |
| 4 | `loan_int_rate` | 0.583 |
| 5 | `interest_burden` | 0.501 |
| 6 | `loan_percent_income` | 0.472 |
| 7 | `credit_score` | 0.196 |

`previous_loan_defaults_on_file` accounts for a mean absolute SHAP value more than **7× greater** than the second most important feature and over **25×** greater than `credit_score` (rank 29 out of 29 features).

### 7.2 Direction of the Effect — The Key Finding

The SHAP beeswarm plot reveals a counterintuitive and consistent pattern across **all 9,000 test applicants**:

- Applicants with **No** prior default on file (`previous_loan_defaults_on_file = No`, shown in **red**) have **strongly positive SHAP values (+0.7 to +1.9)** → the model increases their predicted default probability.
- Applicants with **Yes** (a prior default on file, shown in **blue**) have **strongly negative SHAP values (-10 to -6.5)** → the model dramatically *decreases* their predicted default probability.

This means **having a prior default recorded actually lowers the model's default prediction**, the exact opposite of conventional underwriting intuition. This pattern is also visible in the raw EDA (Figure 4): applicants with no prior default on file default at approximately 45%, while those with a prior default default at nearly 0%.

This raises a critical model-risk question: is this a genuine generalizable behavioural signal (e.g., previously defaulted applicants are more financially cautious when taking new loans), a data-labelling artefact, or a peculiarity of this specific dataset's construction? **SHAP makes this question visible. Aggregate accuracy metrics do not.**

### 7.3 Individual Prediction — Case Study

The SHAP waterfall plot for the single highest-risk applicant in the test set decomposes the model's prediction from a baseline log-odds of **0.049** to a final score of **10.782** (extreme high risk). The five largest contributors for this individual:

1. `loan_to_income_ratio` → **+2.23**
2. `person_home_ownership` → **+2.13**
3. `interest_burden` → **+1.86**
4. `previous_loan_defaults_on_file` → **+1.74**
5. `affordability_ratio` → **+1.44**

Notably, for this individual, the top driver at the population level (`previous_loan_defaults_on_file`) ranks only **4th**. This demonstrates a core principle of SHAP: **global importance ≠ individual importance**. For any given applicant, the set of features that matter most may differ substantially from the population-level ranking — precisely the level of personalised explanation that Regulation B adverse action notices require.

---

## 8. Regulatory & Compliance Implications

This project is directly relevant to the intersection of **model risk management** and **fair lending regulation**:

- **Regulation B / ECOA** (United States): Lenders must provide specific reasons for adverse credit decisions. A model that relies primarily on `previous_loan_defaults_on_file` in a counterintuitive direction would require regulators to independently validate that this effect reflects genuine credit risk and not a data artefact before deployment.
- **SR 11-7 (Federal Reserve Model Risk Guidance)**: Requires that models be conceptually sound and that limitations be clearly understood. The unexplained directionality of the dominant SHAP feature is a textbook example of a model risk flag.
- **IFRS 9** (International Financial Reporting Standard): Requires financial institutions to estimate Expected Credit Loss (ECL) using forward-looking models. SHAP-based explainability frameworks provide the documentation trail required to defend model assumptions to auditors.
- **EU AI Act (2024)**: Classifies credit scoring as a high-risk AI application requiring human oversight and transparency. XAI methods such as SHAP are a direct technical response to this regulatory requirement.

---

## 9. Limitations & Future Work

### Limitations

- **No temporal validation:** The dataset is a single static snapshot. Out-of-time validation (training on earlier periods, testing on later periods) was not conducted and is essential before production deployment.
- **Unexplained dominant variable:** The direction of the `previous_loan_defaults_on_file` effect — prior default lowers predicted risk — is consistent across the entire test population but has not been causally explained. Real-world deployment would require an investigation of the data source before using this variable to inform underwriting policy.
- **No formal fairness analysis:** Disparate impact analysis by gender and other protected characteristics under ECOA was not conducted. This is a mandatory next step before any real-world application.
- **Outlier retention:** Extreme values (age = 144, income = $7.2M) were retained without imputation or removal. A production pipeline would apply systematic outlier treatment.
- **SHAP ≠ causality:** SHAP values reflect what the model has learned to rely on, not necessarily true causal relationships between features and default risk.

### Future Work

- Implement **temporal cross-validation** (walk-forward splits) to simulate real-world deployment conditions.
- Conduct **formal disparate impact analysis** (adverse impact ratio, demographic parity) using the gender variable.
- Extend to **multi-class classification** (on-time, 30-DPD, 60-DPD, 90-DPD) to align with IFRS 9 staging categories.
- Explore **calibration** (Platt scaling, isotonic regression) to convert raw scores to well-calibrated probabilities of default.
- Deploy the best model as a **REST API** (FastAPI + Docker) with a SHAP explanation endpoint.
- Investigate the `previous_loan_defaults_on_file` paradox through **counterfactual analysis** and external dataset validation.

---

## 10. How to Reproduce

```bash
# Option A — Full pipeline via Python script
python src/loan_default_pipeline_custom.py

# Option B — Step-by-step in Jupyter Notebook
jupyter notebook notebooks/loan_default_thesis_pipeline.ipynb
```

All random states are fixed at `RS = 42`. The full pipeline runs end-to-end in approximately **25–40 minutes** on a standard laptop (Optuna tuning dominates runtime). Outputs (figures, tables, model `.pkl` files) are saved automatically to `outputs/`.

---

## 11. References

1. Zhang, X., Zhang, T., Hou, L., Liu, X., Guo, Z., Tian, Y., & Liu, Y. (2025). Data-driven loan default prediction: A machine learning approach for enhancing business process management. *Systems*, *13*(7), 581. https://doi.org/10.3390/systems13070581

2. Melsom, B., Vennerød, C. B., de Lange, P. E., Hjelkrem, L. O., & Westgaard, S. (2022). Explainable artificial intelligence for credit scoring in banking. *Journal of Risk*. https://doi.org/10.21314/JOR.2022.011

3. Xie, S., & Shingadia, T. (2025). Explainable machine learning framework for predicting auto loan defaults. *Risks*, *13*(9), 172. https://doi.org/10.3390/risks13090172

4. Souadda, L. I., Halitim, A. R., Benilles, B., Oliveira, J. M., & Ramos, P. (2025). Optimizing credit risk prediction for peer-to-peer lending using machine learning. *Forecasting*, *7*(3), 35. https://doi.org/10.3390/forecast7030035

5. Jakubik, P., & Teleu, S. (2025). Improving credit risk assessment in uncertain times: Insights from IFRS 9. *Risks*, *13*(2), 38.

6. Al Shiam, S. A., Hasan, M. M., Pantho, M. J., & Nguyen, T. N. (2024). Credit risk prediction using explainable AI. *Journal of Business & Management Studies*, *6*(2), 61–66.

7. Černevičienė, J., & Kabašinskas, A. (2024). Explainable artificial intelligence (XAI) in finance: A systematic literature review. *Artificial Intelligence Review*, *57*(8), 216.

8. Zhi, Z. K., Sin, Y. T., Tan, Y. S. G., & Ng, W. C. (2025). Loan default prediction using machine learning algorithms. *Journal of Informatics and Web Engineering*, *4*(3), 232–244.

9. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, *21*(9), 1263–1284.

10. Felt, G. S. (2024). *Predicting loan default with XGBoost: An examination of strength and application* [Master's thesis]. University of Tartu.

11. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, *30*, 4765–4774.

12. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623–2631.

13. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, *16*, 321–357.

14. Koc, O., Ugur, O., & Kestel, A. S. (2023). The impact of feature selection and transformation on machine learning methods in determining the credit scoring. *arXiv preprint arXiv:2303.05427*.

15. Robinson, N., & Sindhwani, N. (2024). Loan default prediction using machine learning. *2024 11th International Conference on Reliability, Infocom Technologies and Optimization (ICRITO)*, 1–5. IEEE.

16. Lai, L. (2020). Loan default prediction with machine learning techniques. *2020 International Conference on Computer Communication and Network Security (CCNS)*, 5–9. IEEE.

17. Kruthika, M. S., Rajadevi, R., Senthilvadivu, K., Lavanya, J., Deepak, D., & Lavanya, G. (2024). Comparative study of machine learning approaches for loan approval prediction in financial sector. *2024 15th International Conference on Computing Communication and Networking Technologies (ICCCNT)*, 1–6. IEEE.

18. Naik, K. S. (2021). Predicting credit risk for unsecured lending: A machine learning approach. *arXiv preprint arXiv:2110.02206*.

---

## 12. Citation

If you use this work in your research, please cite:

```bibtex
@misc{owusuansah2026explainable,
  author       = {Owusu Ansah, Junior},
  title        = {Explainable Machine Learning for Consumer Loan Default Prediction:
                  A Comparative Model Study with SHAP-Based Interpretability},
  year         = {2026},
  institution  = {New Mexico Highlands University},
  note         = {M.S. Computer Science / MBA Finance Capstone Project},
  url          = {https://github.com/your-username/loan-default-prediction}
}
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built with academic rigour and financial domain expertise.<br>
  Designed to bridge the gap between model performance and regulatory explainability.</i>
</p>
