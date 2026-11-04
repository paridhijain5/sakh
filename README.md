# Sakh (साख): Alternative Credit Scoring for the Unbanked

**Neural Storm 2026 · Problem statement HI-04 · Track: Human Impact**
**Team:** Vanta · **Member:** Paridhi Jain (solo)

**Live demo:** https://sakh-credit-score.streamlit.app · **Repo:** https://github.com/paridhijain5/sakh

> *Credit standing, written in the language of everyday work.*

A *bahi-khata* is the red cloth ledger Indian shopkeepers use to record who paid, who owes and who can be trusted. Sakh brings that idea to gig workers and farmers who have no formal credit file. It scores them from everyday habits, explains every decision, audits itself for bias and tells applicants exactly how to qualify next.

> **All applicant data and labels in this repository are synthetic.** Sakh is a decision-support prototype, not a real lending system. Do not use its scores to approve, price or deny real credit.

---

## Problem and approach

HI-04 asks for an ML model that evaluates creditworthiness for rural and gig workers using alternative data (utility bills, mobile usage) instead of formal credit history.

1. **Data.** A deterministic generator creates 5,000 synthetic applicants. Historical repayment gaps and proxy-feature gaps (for example lower UPI inflow for women and rural applicants) are seeded on purpose, so the fairness audit has something real to find.
2. **Models.** A calibrated logistic-regression baseline and a calibrated XGBoost classifier. Gender, region and occupation are **never** model inputs. They are kept only for the fairness audit.
3. **Score.** Calibrated repayment probability is mapped to a 300–900 score and an illustrative loan tier.
4. **Explainability.** Local SHAP explanations are turned into plain-language reasons.
5. **Recourse.** A bounded search over actionable habits only (on-time bill rate, bill streak, recharge regularity, savings-like balance) finds the smallest realistic change that moves an applicant up a band.
6. **Fairness.** Selection rate, true-positive rate and false-positive rate by gender, region and occupation. A Fairlearn ExponentiatedGradient mitigation is compared with the unmitigated model, and a proxy-feature ablation retrains without two proxy-like features.

## Screenshots

![Applicant score and explanation](screenshots/01-applicant.png)
![Recourse plan](screenshots/02-recourse.png)
![Lender view](screenshots/03-lender.png)
![Fairness audit](screenshots/04-fairness.png)

## Demo walkthrough

- **Applicant:** behavioural sliders, score gauge, suggested tier, plain-language SHAP reasons and a "How to improve" plan.
- **Lender view:** a filterable sample of held-out synthetic applicants with score, tier and top signal.
- **Fairness:** group-level approval and error-rate charts before and after mitigation, plus an opt-in proxy-feature ablation.
- **Privacy:** consent checkbox, privacy notice, a visible synthetic-data stamp, and no storage of slider inputs.

The interface is styled like a bahi-khata: parchment paper, indigo ink, vermilion cloth and a brass gauge.

## Tech stack

Python 3.11 · pandas · NumPy · scikit-learn · XGBoost · SHAP · Fairlearn · Plotly · Streamlit · joblib

## Install and run locally (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\src\data_gen.py
python -m src.model
streamlit run app.py
```

Streamlit opens at http://localhost:8501. If the model file is missing, the app trains and saves it on first launch. The first launch of the hosted demo can take a few minutes for the same reason.

## Model performance

Metrics come from the same stratified, held-out 20% test set. Probabilities use sigmoid calibration with three-fold cross-validation. F1, precision and recall use a 0.50 threshold.

| Model | AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| Calibrated logistic regression | 0.922 | 0.930 | 0.908 | 0.953 | 0.889 |
| Calibrated XGBoost | 0.920 | 0.927 | 0.907 | 0.947 | 0.884 |

The two models perform almost identically on this synthetic data. I kept XGBoost as the selected model because it can capture non-linear behaviour patterns and works with SHAP's tree explainer. The logistic-regression baseline shows the data is largely linearly separable.

These figures were captured in a local Python 3.14 environment. The pinned Python 3.11 deployment may differ slightly, and `python -m src.model` regenerates `models/metrics.json` with the installed versions. These are prototype measurements on synthetic data, not lending-performance claims.

**Score mapping:** `round(300 + 600 × calibrated repayment probability)`. Illustrative maximums: 600–699 → ₹15,000; 700–799 → ₹30,000; 800–900 → ₹50,000; below 600 → no suggested amount.

## Fairness results

Demographic parity difference and equalized-odds difference, before → after ExponentiatedGradient mitigation (ε = 0.02):

| Attribute | Demographic parity diff. | Equalized odds diff. |
|---|---|---|
| Gender | 0.042 → 0.043 | 0.035 → 0.027 |
| Region | 0.094 → 0.025 | 0.049 → 0.110 |
| Occupation | 0.145 → 0.059 | 0.226 → 0.404 |

**Honest reading:** mitigation reduced the parity gaps for region and occupation, but the equalized-odds gaps for both got worse. The unmitigated model's accuracy was 0.884. Mitigated accuracy was 0.889 (gender), 0.871 (region) and 0.845 (occupation). Reducing one disparity can worsen another, and none of this establishes fairness.

**Proxy-feature ablation** (retrained without `upi_inflow_avg` and `avg_recharge_amount`): accuracy 0.884 → 0.874 and AUC 0.920 → 0.914. Demographic parity differences were 0.005 (gender), 0.099 (region) and 0.162 (occupation). Equalized-odds differences were 0.097, 0.024 and 0.238. Removing the features helped the gender gap, slightly worsened region and occupation, and cost some performance.

Full group-level numbers are in `models/fairness_metrics.json`.

## Responsible-AI notes and limitations

- The dataset is wholly synthetic and contains designed disparities. No real person, account or credit-bureau record is used.
- Demographic attributes are excluded from the model, but behavioural variables (especially UPI inflow and recharge amount) can act as proxies for access, income, geography or gender.
- A lower fairness gap is not proof of fairness. ExponentiatedGradient is shown as an experimental comparison, not production mitigation.
- SHAP shows associations in the fitted model, not causes. Recourse suggestions are simplified, may not be feasible for everyone, and do not promise a real-world score increase.
- Probabilities, bands and rupee amounts are illustrative. There is no affordability check, identity verification, lender-grade consent workflow, security review, legal review or external validation.
- Before any real use: obtain meaningful informed consent, minimise and protect data, assess disparate impact on representative and lawfully collected data, involve affected communities, validate calibration and stability, add human review and an appeal path, and consult legal and domain experts.

## Repository layout

```
app.py              Streamlit app (Applicant, Lender view, Fairness)
src/data_gen.py     Synthetic data generator
src/model.py        Training, calibration, scoring
src/recourse.py     Bounded recourse search
src/fairness.py     Fairlearn audit and mitigation
data/               Synthetic dataset
models/             Metrics JSON (model file is rebuilt on first run)
.streamlit/         Ledger theme
screenshots/        README images
```

## AI tools used

Claude (Anthropic) was used for planning and code assistance, and GitHub Copilot (agent mode in VS Code) helped implement and test the project. I reviewed and ran all generated code. The data and reported metrics are synthetic.
