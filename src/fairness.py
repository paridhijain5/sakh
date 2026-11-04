"""Group fairness measurements and a Fairlearn mitigation comparison."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score

from .data_gen import DEMOGRAPHIC_COLS


def _measure(y_true, y_pred, groups):
    from fairlearn.metrics import (MetricFrame, demographic_parity_difference,
                                   equalized_odds_difference, false_positive_rate,
                                   selection_rate, true_positive_rate)

    frame = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
        },
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        sensitive_features=np.asarray(groups),
    )
    by_group = {
        str(group): {name: float(value) for name, value in row.items()}
        for group, row in frame.by_group.iterrows()
    }
    return {
        "by_group": by_group,
        "demographic_parity_difference": float(
            demographic_parity_difference(
                np.asarray(y_true), np.asarray(y_pred),
                sensitive_features=np.asarray(groups),
            )
        ),
        "equalized_odds_difference": float(
            equalized_odds_difference(
                np.asarray(y_true), np.asarray(y_pred),
                sensitive_features=np.asarray(groups),
            )
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def run_fairness_audit(bundle):
    """Compare the selected model with Fairlearn ExponentiatedGradient."""
    from fairlearn.reductions import DemographicParity, ExponentiatedGradient
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x_train = bundle["x_train"]
    x_test = bundle["x_test"]
    y_train = bundle["y_train"]
    y_test = bundle["y_test"]
    scaler = StandardScaler().fit(x_train)
    scaled_train = scaler.transform(x_train)
    scaled_test = scaler.transform(x_test)
    all_data = bundle["source_data"]
    before_predictions = (bundle["model"].predict_proba(x_test)[:, 1] >= 0.5).astype(int)
    by_attribute = {}
    for attribute in DEMOGRAPHIC_COLS:
        train_sensitive = all_data.loc[x_train.index, attribute].astype(str)
        test_sensitive = all_data.loc[x_test.index, attribute].astype(str)
        before = _measure(y_test, before_predictions, test_sensitive)
        mitigation = ExponentiatedGradient(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            constraints=DemographicParity(),
            eps=0.02,
            max_iter=25,
        )
        mitigation.fit(scaled_train, y_train, sensitive_features=train_sensitive)
        after_predictions = mitigation.predict(scaled_test, random_state=42)
        after = _measure(y_test, after_predictions, test_sensitive)
        by_attribute[attribute] = {"before": before, "after": after}

    return {
        "method": "Fairlearn ExponentiatedGradient with demographic-parity constraint (eps=0.02)",
        "test_rows": int(len(y_test)),
        "by_attribute": by_attribute,
    }


def proxy_feature_comparison(bundle):
    """Retrain without the two strongest income/digital-access proxy features."""
    from .model import (PROXY_FEATURES, make_boosted_estimator,
                        make_calibrated_model, metric_row)

    features = [name for name in bundle["feature_cols"] if name not in PROXY_FEATURES]
    estimator, model_name = make_boosted_estimator()
    model = make_calibrated_model(estimator)
    model.fit(bundle["x_train"][features], bundle["y_train"])
    probabilities = model.predict_proba(bundle["x_test"][features])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    result = {
        "model": model_name,
        "dropped_features": PROXY_FEATURES,
        "performance": metric_row(bundle["y_test"], probabilities),
        "by_attribute": {},
    }
    source_data = bundle["source_data"]
    for attribute in DEMOGRAPHIC_COLS:
        groups = source_data.loc[bundle["x_test"].index, attribute].astype(str)
        result["by_attribute"][attribute] = _measure(
            bundle["y_test"], predictions, groups
        )
    return result