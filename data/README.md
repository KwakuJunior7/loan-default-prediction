# Data

The raw dataset is **not included** in this repository because it is hosted on Kaggle.

## How to download

1. Go to: https://www.kaggle.com/datasets/udaymalviya/bank-loan-data
2. Click **Download**
3. Place the file in this `data/` folder and rename it to `loan_data.csv`

## Dataset summary

| Property | Value |
|---|---|
| Filename | `loan_data.csv` |
| Rows | 45,000 |
| Columns | 14 |
| Target | `loan_status` (1 = Default, 0 = No Default) |
| Default rate | 22.2% |
| Missing values | None |

## Alternative: Kaggle API

If you have the Kaggle CLI installed:

```bash
pip install kaggle
kaggle datasets download -d udaymalviya/bank-loan-data
unzip bank-loan-data.zip -d data/
```

You will need a `kaggle.json` API key. See https://www.kaggle.com/docs/api for setup.
