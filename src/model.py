"""Training, evaluation, and inference helpers for Sakh."""
from __future__ import annotations

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_gen import (DATA_PATH, DEMOGRAPHIC_COLS, FEATURE_COLS, TARGET_COL,
                       load_or_generate)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "sakh_model.joblib")
METRICS_PATH = os.path.join(ROOT, "models", "metrics.json")
FAIRNESS_PATH = os.path.join(ROOT, "models", "fairness_metrics.json")
PROXY_FEATURES = ["upi_inflow_avg", "avg_recharge_amount"]
SCORE_BANDS = [
    {"name": "Build a track record", "min_score": 300, "max_score": 499, "loan_inr": 0},
    {"name": "Near eligibility", "min_score": 500, "max_score": 599, "loan_inr": 0},
    {"name": "Starter", "min_score": 600, "max_score": 699, "loan_inr": 15000},
    {"name": "Growth", "min_score": 700, "max_score": 799, "loan_inr": 30000},
    {"name": "Strong", "min_score": 800, "max_score": 900, "loan_inr": 50000},
]


def make_boosted_estimator():
    """Prefer XGBoost; retain a sklearn fallback for offline/local installs."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        warnings.warn("xgboost is not installed; using sklearn HistGradientBoostingClassifier.")
        return HistGradientBoostingClassifier(
            max_iter=140, learning_rate=0.07, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42,
        ), "HistGradientBoosting (fallback)"
    return XGBClassifier(
        n_estimators=160, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.9, reg_lambda=2.0,
        eval_metric="logloss", tree_method="hist", n_jobs=4,
        random_state=42,
    ), "XGBoost"


def make_calibrated_model(estimator):
    return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=3)


def metric_row(y_true, probabilities):
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
    }


def fit_comparison(data: pd.DataFrame, feature_cols=None):
    """Fit calibrated logistic and boosted classifiers on an identical split."""
    feature_cols = list(feature_cols or FEATURE_COLS)
    x = data[feature_cols].astype(float)
    y = data[TARGET_COL].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    logistic = make_calibrated_model(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    )
    logistic.fit(x_train, y_train)

    boosted_estimator, boosted_name = make_boosted_estimator()
    boosted = make_calibrated_model(boosted_estimator)
    boosted.fit(x_train, y_train)

    metrics = {
        "Logistic regression (calibrated)": metric_row(
            y_test, logistic.predict_proba(x_test)[:, 1]
        ),
        f"{boosted_name} (calibrated)": metric_row(
            y_test, boosted.predict_proba(x_test)[:, 1]
        ),
    }
    return {
        "model": boosted,
        "baseline": logistic,
        "model_name": boosted_name,
        "feature_cols": feature_cols,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "test_df": data.loc[x_test.index].copy(),
        "source_data": data,
        "metrics": metrics,
        "synthetic": True,
    }


def score_from_probability(probability: float) -> int:
    return int(np.clip(round(300 + 600 * float(probability)), 300, 900))


def loan_for_score(score: int) -> int:
    for band in reversed(SCORE_BANDS):
        if score >= band["min_score"]:
            return band["loan_inr"]
    return 0


def band_for_score(score: int) -> str:
    for band in SCORE_BANDS:
        if band["min_score"] <= score <= band["max_score"]:
            return band["name"]
    return SCORE_BANDS[-1]["name"]


def predict_applicant(bundle, values):
    row = pd.DataFrame([{name: values[name] for name in bundle["feature_cols"]}])
    probability = float(bundle["model"].predict_proba(row)[:, 1][0])
    score = score_from_probability(probability)
    return {
        "probability": probability,
        "score": score,
        "band": band_for_score(score),
        "loan_inr": loan_for_score(score),
    }


def explain_applicant(bundle, values):
    """Return local attributions (SHAP when installed, local deltas otherwise)."""
    feature_cols = bundle["feature_cols"]
    row = pd.DataFrame([{name: values[name] for name in feature_cols}])
    background = bundle["x_train"][feature_cols].sample(
        min(32, len(bundle["x_train"])), random_state=42
    )
    try:
        import shap

        explainer = shap.Explainer(
            lambda frame: bundle["model"].predict_proba(
                pd.DataFrame(frame, columns=feature_cols)
            )[:, 1],
            background,
            algorithm="permutation",
        )
        explanation = explainer(row, max_evals=2 * len(feature_cols) + 1)
        values_by_feature = np.asarray(explanation.values).reshape(-1)
        method = "SHAP"
    except ImportError:
        baseline = bundle["model"].predict_proba(background)[:, 1].mean()
        original = float(bundle["model"].predict_proba(row)[:, 1][0])
        values_by_feature = []
        for feature in feature_cols:
            changed = row.copy()
            changed[feature] = background[feature].median()
            values_by_feature.append(
                original - float(bundle["model"].predict_proba(changed)[:, 1][0])
            )
        values_by_feature = np.asarray(values_by_feature)
        method = "local feature impact"
        if not np.isfinite(baseline):
            raise ValueError("Could not calculate the explanation baseline.")

    ranked = sorted(
        zip(feature_cols, values_by_feature),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    return method, ranked


def train_and_save():
    data = load_or_generate(DATA_PATH)
    bundle = fit_comparison(data)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    metrics_payload = {
        "dataset": "Synthetic only",
        "rows": int(len(data)),
        "repayment_rate": float(data[TARGET_COL].mean()),
        "test_rows": int(len(bundle["y_test"])),
        "features": bundle["feature_cols"],
        "demographics_used_as_features": False,
        "models": bundle["metrics"],
        "selected_model": bundle["model_name"],
        "score_formula": "round(300 + 600 * calibrated_repayment_probability)",
        "score_bands": SCORE_BANDS,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    from .fairness import run_fairness_audit

    fairness = run_fairness_audit(bundle)
    with open(FAIRNESS_PATH, "w", encoding="utf-8") as file:
        json.dump(fairness, file, indent=2)
    return bundle, metrics_payload, fairness


def load_or_train():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    bundle, _, _ = train_and_save()
    return bundle


def main():
    bundle, metrics, fairness = train_and_save()
    print("Sakh model results (synthetic data; held-out 20% test set)")
    print(f"Rows: {metrics['rows']:,}; test rows: {metrics['test_rows']:,}")
    print(pd.DataFrame(metrics["models"]).T.round(3).to_string())
    print(f"\nSelected model: {bundle['model_name']}")
    for attribute, result in fairness["by_attribute"].items():
        print(
            f"{attribute}: demographic parity difference "
            f"{result['before']['demographic_parity_difference']:.3f} -> "
            f"{result['after']['demographic_parity_difference']:.3f}; "
            f"equalized odds difference "
            f"{result['before']['equalized_odds_difference']:.3f} -> "
            f"{result['after']['equalized_odds_difference']:.3f}"
        )
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved fairness audit: {FAIRNESS_PATH}")


if __name__ == "__main__":
    main()
