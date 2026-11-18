# Data

The dataset is **not included** in this repository. Download it from Kaggle and save it here as `loan_data.csv`.

## Download

- Page: https://www.kaggle.com/datasets/udaymalviya/bank-loan-data
- Or with the Kaggle CLI (needs a `kaggle.json` API key):

```bash
kaggle datasets download -d udaymalviya/bank-loan-data
unzip bank-loan-data.zip -d data/
```

## What the data is

| Property | Value |
|---|---|
| Rows × columns | 45,000 × 14 |
| Target | `loan_status` (treated here as 1 = default, 0 = no default; see below) |
| Positive rate | 22.2% |
| Missing values | None |
| Origin | **Synthetic.** The Kaggle page says so; it is a re-upload of [Loan Approval Classification Data](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data), which was generated with SMOTENC from the older [Credit Risk dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset). |

## About the label

The two Kaggle pages describe `loan_status` differently: the re-upload says 1 = repaid, and the original synthetic version says 1 = approved. The data itself behaves like the default label of the underlying Credit Risk dataset: `loan_status = 1` comes with higher interest rates (12.9% vs 10.5%), a higher loan-to-income ratio (0.20 vs 0.12) and lower income ($59.9k vs $86.2k). This project therefore treats 1 as default, and says so wherever results are reported.

## Known artefacts (checked by the pipeline's data audit)

- Every applicant with `previous_loan_defaults_on_file = Yes` (22,858 rows, 51%) has `loan_status = 0`. This is a generation rule, not behaviour, so the README reports results with and without that feature.
- `credit_score` is unrelated to the target: the positive rate is 21–22% in every credit-score quintile.
- 7 rows have `person_age` above 100 (maximum 144); the pipeline drops them.
