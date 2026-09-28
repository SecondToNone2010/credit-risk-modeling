# Credit risk modeling case

This is a personal credit-risk project built around the UCI **Default of Credit Card Clients** dataset (30,000 accounts). I started it because I wanted to understand the pieces around a model, not just fit a classifier and report accuracy.

The current version keeps the workflow fairly small:

- basic repayment and utilization features
- development / validation / untouched holdout split
- logistic PD model as the main model
- calibrated gradient-boosting model as a challenger
- AUC, Gini, KS and Brier score
- simple risk-grade backtesting
- score PSI between development and holdout
- illustrative PD/LGD/EAD stress scenarios

I intentionally do not call this an IFRS 9 or Basel model. The dataset is old, non-Vietnamese, has no realised recoveries or contractual EAD, and does not contain a real time series for out-of-time validation. The LGD/CCF numbers in the stress section are just sensitivity assumptions.

## Data

Download the public UCI dataset **Default of Credit Card Clients** and place the original `.xls` file somewhere locally. The script also accepts a cleaned CSV using the column names in `data/README.md`.

Dataset reference: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

I do not commit the raw dataset to this repository.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python credit_risk.py --data data/default_of_credit_card_clients.xls
```

The script writes a few small tables to `output/`:

- `model_metrics.csv`
- `risk_grade_backtest.csv`
- `stability.csv`
- `stress_scenarios.csv`

## What I would improve next

The biggest limitation is the lack of true observation dates. A better version would use a real bank-style performance window and an out-of-time sample. I would also want observed LGD/EAD data before treating the loss section as anything more than a sensitivity exercise.
