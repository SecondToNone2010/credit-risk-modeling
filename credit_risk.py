from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET = "default_next_month"

BASE_COLUMNS = [
    "limit_balance",
    "pay_status_1", "pay_status_2", "pay_status_3", "pay_status_4", "pay_status_5", "pay_status_6",
    "bill_amount_1", "bill_amount_2", "bill_amount_3", "bill_amount_4", "bill_amount_5", "bill_amount_6",
    "payment_amount_1", "payment_amount_2", "payment_amount_3", "payment_amount_4", "payment_amount_5", "payment_amount_6",
]


def load_data(path: Path) -> pd.DataFrame:
    """Read either the original UCI xls or a pre-cleaned csv."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    # UCI's file has a title row above the actual header.
    df = pd.read_excel(path, header=1)
    rename = {
        "ID": "customer_id",
        "LIMIT_BAL": "limit_balance",
        "SEX": "sex",
        "EDUCATION": "education",
        "MARRIAGE": "marriage",
        "AGE": "age",
        "PAY_0": "pay_status_1",
        "PAY_2": "pay_status_2",
        "PAY_3": "pay_status_3",
        "PAY_4": "pay_status_4",
        "PAY_5": "pay_status_5",
        "PAY_6": "pay_status_6",
        "BILL_AMT1": "bill_amount_1",
        "BILL_AMT2": "bill_amount_2",
        "BILL_AMT3": "bill_amount_3",
        "BILL_AMT4": "bill_amount_4",
        "BILL_AMT5": "bill_amount_5",
        "BILL_AMT6": "bill_amount_6",
        "PAY_AMT1": "payment_amount_1",
        "PAY_AMT2": "payment_amount_2",
        "PAY_AMT3": "payment_amount_3",
        "PAY_AMT4": "payment_amount_4",
        "PAY_AMT5": "payment_amount_5",
        "PAY_AMT6": "payment_amount_6",
        "default payment next month": TARGET,
    }
    return df.rename(columns=rename)


def engineer_features(df: pd.DataFrame):
    out = df.copy()
    pay = [f"pay_status_{i}" for i in range(1, 7)]
    bills = [f"bill_amount_{i}" for i in range(1, 7)]
    payments = [f"payment_amount_{i}" for i in range(1, 7)]

    limit_bal = out["limit_balance"].clip(lower=1)
    positive_bills = out[bills].clip(lower=0)
    positive_payments = out[payments].clip(lower=0)

    out["max_delinquency"] = out[pay].max(axis=1)
    out["months_delinquent"] = (out[pay] > 0).sum(axis=1)
    out["avg_utilization"] = positive_bills.div(limit_bal, axis=0).clip(0, 3).mean(axis=1)
    out["max_utilization"] = positive_bills.div(limit_bal, axis=0).clip(0, 3).max(axis=1)
    out["avg_payment_to_bill"] = (
        positive_payments.to_numpy() / np.maximum(positive_bills.to_numpy(), 1)
    ).clip(0, 5).mean(axis=1)
    out["zero_payment_months"] = (positive_payments <= 0).sum(axis=1)
    out["bill_change_3m"] = (
        (positive_bills[bills[0]] - positive_bills[bills[2]]) / limit_bal
    ).clip(-3, 3)
    out["payment_change_3m"] = (
        (positive_payments[payments[0]] - positive_payments[payments[2]]) / limit_bal
    ).clip(-3, 3)

    features = BASE_COLUMNS + [
        "max_delinquency",
        "months_delinquent",
        "avg_utilization",
        "max_utilization",
        "avg_payment_to_bill",
        "zero_payment_months",
        "bill_change_3m",
        "payment_change_3m",
    ]
    return out, features


def split_data(df: pd.DataFrame):
    dev, temp = train_test_split(
        df, test_size=0.40, stratify=df[TARGET], random_state=SEED
    )
    val, holdout = train_test_split(
        temp, test_size=0.50, stratify=temp[TARGET], random_state=SEED
    )
    return dev, val, holdout


def fit_logistic(dev, features):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.3, max_iter=2000, random_state=SEED)),
    ])
    model.fit(dev[features], dev[TARGET])
    return model


def fit_challenger(dev, features):
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=60,
            l2_regularization=1.0,
            random_state=SEED,
        )),
    ])
    model = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    model.fit(dev[features], dev[TARGET])
    return model


def ks_statistic(y, pd_hat):
    frame = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_hat)}).sort_values("pd")
    bad = frame["y"].sum()
    good = len(frame) - bad
    if bad == 0 or good == 0:
        return np.nan
    frame["cum_bad"] = frame["y"].cumsum() / bad
    frame["cum_good"] = (1 - frame["y"]).cumsum() / good
    return float((frame["cum_bad"] - frame["cum_good"]).abs().max())


def model_metrics(name, sample, y, pd_hat):
    auc = roc_auc_score(y, pd_hat)
    return {
        "model": name,
        "sample": sample,
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y, pd_hat),
        "brier": brier_score_loss(y, pd_hat),
        "observed_default_rate": float(np.mean(y)),
        "mean_predicted_pd": float(np.mean(pd_hat)),
    }


def score_from_pd(pd_hat, base_score=600, pdo=20, good_bad_odds=50):
    pd_hat = np.clip(np.asarray(pd_hat), 1e-6, 1 - 1e-6)
    odds_good_bad = (1 - pd_hat) / pd_hat
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(good_bad_odds)
    return offset + factor * np.log(odds_good_bad)


def psi(expected, actual, bins=10):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    e = pd.Series(pd.cut(expected, edges, include_lowest=True)).value_counts(
        normalize=True, sort=False
    )
    a = pd.Series(pd.cut(actual, edges, include_lowest=True)).value_counts(
        normalize=True, sort=False
    )
    e = e.clip(lower=1e-6)
    a = a.clip(lower=1e-6)
    return float(((a - e) * np.log(a / e)).sum())


def risk_grade_table(y, pd_hat, dev_pd):
    edges = np.unique(np.quantile(dev_pd, [0, .2, .4, .6, .8, 1]))
    edges[0], edges[-1] = -np.inf, np.inf
    labels = [f"G{i}" for i in range(1, len(edges))]
    frame = pd.DataFrame({"actual": np.asarray(y), "pd": np.asarray(pd_hat)})
    frame["grade"] = pd.cut(
        frame["pd"], bins=edges, labels=labels, include_lowest=True
    )
    return frame.groupby("grade", observed=True).agg(
        accounts=("actual", "size"),
        observed_default_rate=("actual", "mean"),
        mean_predicted_pd=("pd", "mean"),
    ).reset_index()


def stress_test(df, pd_hat):
    # The public data has no realised LGD/EAD, so these are only sensitivity assumptions.
    bill = df["bill_amount_1"].clip(lower=0).to_numpy()
    limit_bal = df["limit_balance"].clip(lower=0).to_numpy()
    undrawn = np.maximum(limit_bal - bill, 0)

    scenarios = [
        ("base", 1.0, 0.45, 0.40),
        ("adverse", 1.5, 0.55, 0.55),
        ("severe", 2.5, 0.65, 0.70),
    ]
    rows = []
    base_odds = pd_hat / np.clip(1 - pd_hat, 1e-8, None)

    for name, pd_mult, lgd, ccf in scenarios:
        stressed_odds = base_odds * pd_mult
        stressed_pd = stressed_odds / (1 + stressed_odds)
        ead = bill + ccf * undrawn
        ecl = stressed_pd * lgd * ead
        rows.append({
            "scenario": name,
            "mean_pd": float(np.mean(stressed_pd)),
            "lgd": lgd,
            "ccf": ccf,
            "portfolio_ecl": float(np.sum(ecl)),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="UCI .xls file or a cleaned csv")
    parser.add_argument("--out", default="output")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_data(Path(args.data))
    df, features = engineer_features(raw)
    dev, val, holdout = split_data(df)

    logistic = fit_logistic(dev, features)
    challenger = fit_challenger(dev, features)

    dev_pd = logistic.predict_proba(dev[features])[:, 1]
    val_pd = logistic.predict_proba(val[features])[:, 1]
    hold_pd = logistic.predict_proba(holdout[features])[:, 1]
    challenger_hold_pd = challenger.predict_proba(holdout[features])[:, 1]

    metrics = pd.DataFrame([
        model_metrics("logistic", "development", dev[TARGET], dev_pd),
        model_metrics("logistic", "validation", val[TARGET], val_pd),
        model_metrics("logistic", "holdout", holdout[TARGET], hold_pd),
        model_metrics("calibrated_hgb", "holdout", holdout[TARGET], challenger_hold_pd),
    ])
    metrics.to_csv(out_dir / "model_metrics.csv", index=False)

    grades = risk_grade_table(holdout[TARGET], hold_pd, dev_pd)
    grades.to_csv(out_dir / "risk_grade_backtest.csv", index=False)

    dev_score = score_from_pd(dev_pd)
    hold_score = score_from_pd(hold_pd)
    pd.DataFrame([{"score_psi_holdout_vs_dev": psi(dev_score, hold_score)}]).to_csv(
        out_dir / "stability.csv", index=False
    )

    stress_test(holdout, hold_pd).to_csv(
        out_dir / "stress_scenarios.csv", index=False
    )

    print(metrics.to_string(index=False))
    print(f"\nScore PSI (holdout vs development): {psi(dev_score, hold_score):.4f}")
    print(f"Outputs written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
