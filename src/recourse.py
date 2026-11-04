"""Greedy, bounded recourse using only applicant-actionable behaviours."""
from __future__ import annotations

from .model import loan_for_score, predict_applicant

ACTIONABLE_STEPS = {
    "bill_on_time_rate": 1 / 12,
    "bill_streak_months": 1,
    "recharge_regularity": 1 / 12,
    "min_balance_ratio": 0.05,
}
BOUNDS = {
    "bill_on_time_rate": (0.25, 1.0),
    "bill_streak_months": (0, 12),
    "recharge_regularity": (0.2, 1.0),
    "min_balance_ratio": (0.0, 1.0),
}


def _next_target(score):
    return next((target for target in (600, 700, 800) if score < target), None)


def _action_text(feature, steps):
    if feature == "bill_on_time_rate":
        return f"pay utility bills on time for {steps} more month(s)"
    if feature == "bill_streak_months":
        return f"build the on-time bill streak by {steps} month(s)"
    if feature == "recharge_regularity":
        return f"keep mobile recharges regular for {steps} more month(s)"
    return f"maintain a small savings buffer for {steps} more month(s)"


def recommend_recourse(bundle, applicant):
    """Find a short greedy path to the next score tier without changing identity."""
    before = predict_applicant(bundle, applicant)
    target = _next_target(before["score"])
    if target is None:
        return {
            "reachable": False,
            "before": before,
            "after": before,
            "message": "You are already in the highest score band.",
            "actions": [],
        }

    changed = dict(applicant)
    used = {feature: 0 for feature in ACTIONABLE_STEPS}
    for _ in range(80):
        state = predict_applicant(bundle, changed)
        if state["score"] >= target:
            break
        candidates = []
        for feature, step in ACTIONABLE_STEPS.items():
            low, high = BOUNDS[feature]
            candidate_value = min(high, max(low, float(changed[feature]) + step))
            if candidate_value <= float(changed[feature]):
                continue
            candidate = dict(changed)
            candidate[feature] = candidate_value
            candidate_score = predict_applicant(bundle, candidate)["score"]
            candidates.append((candidate_score, feature, candidate))
        if not candidates:
            break
        _, feature, changed = max(
            candidates, key=lambda item: (item[0], -used[item[1]])
        )
        used[feature] += 1

    after = predict_applicant(bundle, changed)
    actions = [
        _action_text(feature, count)
        for feature, count in used.items()
        if count
    ]
    reachable = after["score"] >= target
    if reachable:
        message = (
            f"To move from {before['score']} to {after['score']} (the next band, "
            f"which suggests up to ₹{loan_for_score(after['score']):,}), try: "
            + "; ".join(actions)
            + ". This is a model estimate, not a lending decision."
        )
    else:
        message = (
            f"The next score band starts at {target}. The actionable changes tested "
            f"could move the estimate from {before['score']} to {after['score']}, "
            "but did not reach that band. No identity or demographic attributes "
            "were changed."
        )
    return {
        "reachable": reachable,
        "target_score": target,
        "before": before,
        "after": after,
        "message": message,
        "actions": actions,
        "changes": {
            feature: changed[feature]
            for feature, count in used.items()
            if count
        },
    }
