"""
Sakh - synthetic applicant data generator.

ALL DATA PRODUCED HERE IS SYNTHETIC. No real person is represented.
Historical bias is deliberately baked in so the fairness audit has
something real to find:
  * women and rural applicants have lower UPI inflow (proxy-feature gap)
  * rural applicants have lower on-time bill rates (payment infrastructure)
  * women and rural applicants have slightly lower repayment labels
    even at identical behaviour (historical lending / shock exposure bias)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

SEED = 42
N_DEFAULT = 5000
TARGET_REPAY_RATE = 0.78

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "applicants.csv")

OCCUPATIONS = ["Gig driver", "Delivery partner", "Farmer",
               "Street vendor", "Domestic/informal worker", "Small shop owner"]
REGIONS = ["Urban", "Semi-urban", "Rural"]

# Columns used by other modules
DEMOGRAPHIC_COLS = ["gender", "region", "occupation"]      # audit only, NEVER model inputs
FEATURE_COLS = [
    "bill_on_time_rate",      # share of last 12 utility bills paid on time (0-1)
    "bill_streak_months",     # longest on-time bill streak in last 12 months (0-12)
    "recharge_regularity",    # how regularly mobile is recharged (0-1)
    "avg_recharge_amount",    # average recharge, INR
    "upi_inflow_avg",         # average monthly UPI money received, INR
    "upi_inflow_cv",          # coefficient of variation of monthly UPI inflow
    "income_volatility",      # income volatility index (0-1)
    "seasonal_swing",         # (peak month - trough month) / mean inflow
    "active_months",          # months active on platform / in this work
    "min_balance_ratio",      # share of months balance stayed above a small buffer (0-1)
    "informal_loan_burden",   # informal loan repayments as share of monthly income
]
TARGET_COL = "repaid"         # 1 = repaid, 0 = defaulted

# Occupation-level assumptions
OCC_BY_REGION = {
    #            gig   deliv  farmer vendor domestic shop
    "Urban":      [0.25, 0.30, 0.02, 0.15, 0.18, 0.10],
    "Semi-urban": [0.15, 0.15, 0.15, 0.20, 0.15, 0.20],
    "Rural":      [0.05, 0.05, 0.45, 0.12, 0.13, 0.20],
}
P_FEMALE = {"Gig driver": 0.08, "Delivery partner": 0.12, "Farmer": 0.35,
            "Street vendor": 0.40, "Domestic/informal worker": 0.80, "Small shop owner": 0.30}
BASE_INCOME = {"Gig driver": 22000, "Delivery partner": 18000, "Farmer": 14000,
               "Street vendor": 12000, "Domestic/informal worker": 9000, "Small shop owner": 20000}
CV_BASE = {"Gig driver": 0.35, "Delivery partner": 0.35, "Farmer": 0.60,
           "Street vendor": 0.50, "Domestic/informal worker": 0.30, "Small shop owner": 0.35}
SWING_BASE = {"Gig driver": 0.30, "Delivery partner": 0.30, "Farmer": 1.10,
              "Street vendor": 0.40, "Domestic/informal worker": 0.20, "Small shop owner": 0.30}
TENURE_MEAN = {"Gig driver": 18, "Delivery partner": 14, "Farmer": 60,
               "Street vendor": 36, "Domestic/informal worker": 40, "Small shop owner": 40}


def _z(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_applicants(n: int = N_DEFAULT, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- demographics (audit only) ----
    region = rng.choice(REGIONS, size=n, p=[0.35, 0.30, 0.35])
    occupation = np.empty(n, dtype=object)
    for r, probs in OCC_BY_REGION.items():
        idx = np.where(region == r)[0]
        occupation[idx] = rng.choice(OCCUPATIONS, size=len(idx), p=probs)
    p_f = pd.Series(occupation).map(P_FEMALE).to_numpy()
    female = (rng.random(n) < p_f).astype(float)
    gender = np.where(female == 1, "Female", "Male")
    remote = pd.Series(region).map({"Urban": 0.0, "Semi-urban": 0.5, "Rural": 1.0}).to_numpy()

    # ---- hidden latent traits ----
    disc = rng.normal(size=n)                                   # financial discipline
    stab = 0.4 * disc + np.sqrt(1 - 0.16) * rng.normal(size=n)  # income stability

    occ = pd.Series(occupation)
    base_income = occ.map(BASE_INCOME).to_numpy()
    income = (base_income * np.exp(0.25 * rng.normal(size=n) + 0.10 * stab)
              * (1 - 0.12 * female) * (1 - 0.08 * remote))      # baked-in income gap

    # ---- behavioural features (model inputs) ----
    bill = np.clip(0.76 + 0.12 * disc + 0.04 * stab + rng.normal(0, 0.06, n) - 0.04 * remote, 0.25, 1.0)
    streak = np.clip(np.round(12 * bill ** 2 * rng.uniform(0.55, 1.15, n)), 0, 12).astype(int)
    recharge_reg = np.clip(0.74 + 0.11 * disc + 0.05 * stab + rng.normal(0, 0.08, n), 0.2, 1.0)
    avg_recharge = np.clip(np.round(110 + 0.006 * income + rng.normal(0, 35, n)), 49, 799).astype(int)

    digital_share = np.clip(rng.normal(0.72, 0.10, n) - 0.10 * remote - 0.03 * female, 0.25, 0.95)
    upi_avg = np.round(income * digital_share).astype(int)
    cv = np.clip(occ.map(CV_BASE).to_numpy() * np.exp(-0.25 * stab + rng.normal(0, 0.2, n)), 0.08, 1.5)
    vol = np.clip(0.5 * cv + 0.08 + rng.normal(0, 0.06, n), 0.02, 1.0)
    swing = np.clip(occ.map(SWING_BASE).to_numpy() * np.exp(rng.normal(0, 0.3, n) - 0.10 * stab), 0.05, 2.5)
    active = np.clip(np.round(rng.exponential(occ.map(TENURE_MEAN).to_numpy()) + 3), 3, 120).astype(int)
    min_bal = np.clip(0.38 + 0.14 * disc + 0.06 * stab + 0.05 * np.log(income / 15000)
                      + rng.normal(0, 0.10, n), 0.0, 1.0)
    has_loan = rng.random(n) < np.clip(0.45 + 0.10 * remote - 0.05 * disc, 0.10, 0.85)
    burden = has_loan * np.clip(0.12 * np.exp(-0.3 * disc + 0.5 * rng.normal(size=n)), 0.0, 0.8)

    # ---- target: latent function + noise + historical bias ----
    core = (0.90 * _z(bill) + 0.30 * _z(streak) + 0.45 * _z(recharge_reg)
            + 0.25 * _z(np.log(upi_avg)) + 0.10 * _z(np.log(avg_recharge))
            - 0.55 * _z(cv) - 0.25 * _z(vol) - 0.12 * _z(swing)
            + 0.30 * _z(np.log(active)) + 0.45 * _z(min_bal) - 0.65 * _z(burden)
            + 0.25 * disc                      # hidden trait the model can't see
            - 0.25 * female - 0.20 * remote    # baked-in historical label bias
            + rng.normal(0, 0.6, n))
    lo, hi = -10.0, 10.0                        # calibrate intercept to hit target repay rate
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(core + mid).mean() < TARGET_REPAY_RATE:
            lo = mid
        else:
            hi = mid
    repaid = (rng.random(n) < _sigmoid(core + (lo + hi) / 2)).astype(int)

    df = pd.DataFrame({
        "applicant_id": [f"SK-{i + 1:05d}" for i in range(n)],
        "gender": gender, "region": region, "occupation": occupation,
        "bill_on_time_rate": bill.round(3),
        "bill_streak_months": streak,
        "recharge_regularity": recharge_reg.round(3),
        "avg_recharge_amount": avg_recharge,
        "upi_inflow_avg": upi_avg,
        "upi_inflow_cv": cv.round(3),
        "income_volatility": vol.round(3),
        "seasonal_swing": swing.round(3),
        "active_months": active,
        "min_balance_ratio": min_bal.round(3),
        "informal_loan_burden": burden.round(3),
        TARGET_COL: repaid,
    })
    return df


def load_or_generate(path: str = DATA_PATH) -> pd.DataFrame:
    """Used by the app: load the CSV, or regenerate it if missing."""
    if os.path.exists(path):
        return pd.read_csv(path)
    df = generate_applicants()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def print_summary(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 130)
    pd.set_option("display.max_columns", 30)
    print("=" * 70)
    print(" SAKH - SYNTHETIC DATASET (no real individuals; for demo use only)")
    print("=" * 70)
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(f"Overall repayment rate: {df[TARGET_COL].mean():.1%}   "
          f"Default rate: {1 - df[TARGET_COL].mean():.1%}\n")
    for col in DEMOGRAPHIC_COLS:
        g = df.groupby(col).agg(n=(TARGET_COL, "size"),
                                repaid_rate=(TARGET_COL, "mean"),
                                mean_upi_inflow=("upi_inflow_avg", "mean"),
                                mean_bill_rate=("bill_on_time_rate", "mean")).round(3)
        print(f"--- by {col} ---\n{g}\n")
    print("--- model input features ---")
    print(df[FEATURE_COLS].describe().T[["mean", "std", "min", "max"]].round(2), "\n")
    f, m = df[df.gender == "Female"], df[df.gender == "Male"]
    ru, ur = df[df.region == "Rural"], df[df.region == "Urban"]
    print("--- baked-in bias check (this is what the audit should find) ---")
    print(f"Repayment gap  Female vs Male : {f[TARGET_COL].mean() - m[TARGET_COL].mean():+.3f}")
    print(f"Repayment gap  Rural vs Urban : {ru[TARGET_COL].mean() - ur[TARGET_COL].mean():+.3f}")
    print(f"UPI inflow gap Female vs Male : {f.upi_inflow_avg.mean() - m.upi_inflow_avg.mean():+,.0f} INR")
    print(f"UPI inflow gap Rural vs Urban : {ru.upi_inflow_avg.mean() - ur.upi_inflow_avg.mean():+,.0f} INR")


if __name__ == "__main__":
    data = generate_applicants()
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    data.to_csv(DATA_PATH, index=False)
    print(f"Saved -> {DATA_PATH}\n")
    print_summary(data)