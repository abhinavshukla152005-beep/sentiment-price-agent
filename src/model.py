"""Walk-forward (expanding window) evaluation of the n+1 return model.

No shuffling, no random splits: for every test date the model is refit on strictly earlier
dates, so a prediction never sees the bar it is predicting. Reported metrics are compared
against two naive baselines so a score is interpretable rather than just "high".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, roc_auc_score

from .features import feature_columns

MIN_TRAIN_ROWS = 40
MIN_TRAIN_DATES = 5


def _design(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    frame = panel.dropna(subset=["target_up", "target_ret"]).copy()
    for column in features:
        if column not in frame:
            frame[column] = 0.0
    frame[features] = frame[features].astype(float).replace([np.inf, -np.inf], np.nan)
    frame[features] = frame[features].fillna(frame[features].median(numeric_only=True)).fillna(0.0)
    return frame


def walk_forward(panel: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    features = feature_columns(sources)
    frame = _design(panel, features)
    dates = sorted(frame["date"].unique())

    predictions = []
    for split in range(MIN_TRAIN_DATES, len(dates)):
        test_date = dates[split]
        train = frame[frame["date"] < test_date]
        if len(train) < MIN_TRAIN_ROWS or train["target_up"].nunique() < 2:
            continue
        test = frame[frame["date"] == test_date]
        if test.empty:
            continue

        classifier = GradientBoostingClassifier(
            n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0
        )
        regressor = GradientBoostingRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0
        )
        classifier.fit(train[features], train["target_up"])
        regressor.fit(train[features], train["target_ret"])

        prob_up = classifier.predict_proba(test[features])[:, 1]
        expected = regressor.predict(test[features])

        block = test[["date", "ticker", "close", "target_ret", "target_up"]].copy()
        block["prob_up"] = prob_up
        block["expected_ret"] = expected
        predictions.append(block)

    if not predictions:
        return pd.DataFrame(columns=["date", "ticker", "close", "target_ret", "target_up",
                                     "prob_up", "expected_ret"])
    return pd.concat(predictions, ignore_index=True)


def metrics(results: pd.DataFrame) -> dict:
    if results.empty:
        return {"n": 0}

    truth = results["target_up"].to_numpy()
    prob = results["prob_up"].to_numpy()
    predicted_up = (prob >= 0.5).astype(float)

    accuracy = float((predicted_up == truth).mean())
    try:
        auc = float(roc_auc_score(truth, prob))
        balanced = float(balanced_accuracy_score(truth, predicted_up))
    except ValueError:
        auc = float("nan")
        balanced = float("nan")

    majority = float(max(truth.mean(), 1 - truth.mean()))
    naive_mae = float(mean_absolute_error(results["target_ret"], np.zeros(len(results))))
    model_mae = float(mean_absolute_error(results["target_ret"], results["expected_ret"]))
    corr = float(np.corrcoef(results["expected_ret"], results["target_ret"])[0, 1]) if len(results) > 2 else float("nan")

    return {
        "n": int(len(results)),
        "dates": int(results["date"].nunique()),
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced, 4),
        "auc": round(auc, 4),
        "majority_baseline": round(majority, 4),
        "mae_model": round(model_mae, 5),
        "mae_naive": round(naive_mae, 5),
        "ret_corr": round(corr, 4),
    }


def by_exchange(results: pd.DataFrame, exchange_of: dict[str, str]) -> dict:
    if results.empty:
        return {}
    frame = results.copy()
    frame["exchange"] = frame["ticker"].map(exchange_of)
    return {name: metrics(group) for name, group in frame.groupby("exchange")}


def predict_next_day(panel: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    """Fit on every dated row and score the most recent bar of each ticker: the n+1 call."""
    features = feature_columns(sources)
    frame = _design(panel, features)
    train = frame.dropna(subset=["target_up"])
    if train.empty or train["target_up"].nunique() < 2:
        return pd.DataFrame(columns=["ticker", "prob_up", "expected_ret"])

    # Identical hyperparameters to the walk-forward run, so the number shown for tomorrow
    # comes from the configuration whose out-of-sample behaviour was actually measured.
    classifier = GradientBoostingClassifier(
        n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0
    )
    regressor = GradientBoostingRegressor(
        n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0
    )
    classifier.fit(train[features], train["target_up"])
    regressor.fit(train[features], train["target_ret"])

    latest = (
        panel.sort_values("date")
        .groupby("ticker", as_index=False)
        .tail(1)
        .copy()
    )
    for column in features:
        if column not in latest:
            latest[column] = 0.0
    latest[features] = latest[features].astype(float).replace([np.inf, -np.inf], np.nan)
    latest[features] = latest[features].fillna(train[features].median(numeric_only=True)).fillna(0.0)

    latest["prob_up"] = classifier.predict_proba(latest[features])[:, 1]
    latest["expected_ret"] = regressor.predict(latest[features])
    return latest[["ticker", "date", "close", "prob_up", "expected_ret"]].reset_index(drop=True)
