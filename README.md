# Sakh (साख) — HI-04 Alternative Credit Scoring

**Team:** Vanta · **Participant:** Paridhi Jain (solo) · **Hackathon:** Neural Storm 2026  
**Tagline:** *A clearer path to credit, built from everyday financial habits—not a formal credit file.*

> **All applicant data and labels in this repository are synthetic. Sakh is a decision-support prototype, not a real lending decision.** Do not use its scores to approve, price, or deny real credit.

## Problem and approach

HI-04 asks for an ML model that evaluates creditworthiness for rural and gig workers using alternative data such as utility bills and mobile usage instead of formal credit history. Sakh demonstrates a small end-to-end approach:

1. Generate 5,000 deterministic synthetic applicants, with historical repayment and proxy-feature gaps intentionally included for audit practice.
2. Train a calibrated logistic-regression baseline and a calibrated XGBoost classifier. Gender, region, and occupation are excluded from model inputs and retained only for the fairness audit.
3. Map calibrated repayment probability to a 300–900 score and an illustrative loan tier. Provide local SHAP explanations and a bounded recourse search over bill payment, bill streak, recharge regularity, and savings-like balance only.
4. Measure selection, true-positive, and false-positive rates by gender, region, and occupation. Compare the selected model with Fairlearn ExponentiatedGradient mitigation, and retrain without two proxy-like features to show the accuracy/fairness trade-off.

This is a hackathon demo: synthetic labels are not evidence of real-world creditworthiness or model validity.

## Tech stack

Python 3.11 · pandas · NumPy · scikit-learn · XGBoost · SHAP · Fairlearn · Plotly · Streamlit · joblib

## Install and run locally (Windows PowerShell)

Use Python 3.11, matching the pinned deployment dependencies:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\src\data_gen.py
python -m src.model
streamlit run app.py
```

The generator should print `SAKH - SYNTHETIC DATASET`, 5,000 rows, the repayment rate, and the deliberately seeded demographic/proxy gaps. Training prints a held-out comparison table and fairness gaps, then saves `models/sakh_model.joblib`, `models/metrics.json`, and `models/fairness_metrics.json`. Streamlit opens at `http://localhost:8501`. It trains and saves those artifacts on first run if the model file is absent.

`python -m src.model` uses the full pinned stack (including XGBoost and Fairlearn). If XGBoost is unavailable during local development, the code uses scikit-learn HistGradientBoosting as a clearly named fallback; the public deployment installs XGBoost from `requirements.txt`.

## Model performance

Metrics are calculated on the same stratified, held-out 20% test set. Probability calibration uses sigmoid calibration with three-fold cross-validation; F1, precision, and recall use a 0.50 probability threshold. **Results are synthetic-data prototype measurements, not lending-performance claims.**

These captured results came from the available local Python 3.14 environment (XGBoost 3.4.1, scikit-learn 1.9.1, Fairlearn 0.15.0). The pinned Python 3.11 deployment versions may produce slightly different metrics; `python -m src.model` regenerates the JSON results with the installed versions.

| Model | AUC | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|
| Calibrated logistic regression | 0.922 | 0.930 | 0.908 | 0.953 | 0.889 |
| Calibrated XGBoost | 0.920 | 0.927 | 0.907 | 0.947 | 0.884 |

After running the training command, the exact values are recorded in [`models/metrics.json`](models/metrics.json). Score mapping: `round(300 + 600 × calibrated repayment probability)`. Illustrative suggested maximums: 600–699 → ₹15,000; 700–799 → ₹30,000; 800–900 → ₹50,000; below 600 → no suggested amount.

## Fairness results

Fairness is evaluated on the held-out synthetic set, with group selection rate (approval-rate proxy), true-positive rate, and false-positive rate. Demographic parity difference and equalized-odds difference are reported before and after a Fairlearn ExponentiatedGradient demographic-parity constraint (ε=0.02). The app also retrains without `upi_inflow_avg` and `avg_recharge_amount` to expose a proxy-feature ablation.

| Audit attribute | Demographic parity difference (before → after) | Equalized odds difference (before → after) |
|---|---:|---:|
| Gender | 0.042 → 0.043 | 0.035 → 0.027 |
| Region | 0.094 → 0.025 | 0.049 → 0.110 |
| Occupation | 0.145 → 0.059 | 0.226 → 0.404 |

Exact group rates and gaps are saved in [`models/fairness_metrics.json`](models/fairness_metrics.json). Mitigation may reduce one disparity while worsening another or reducing accuracy; it does not establish fairness. Removing proxy-like inputs also does not guarantee that remaining features are unbiased.

The unmitigated selected model's test accuracy was 0.884. Mitigated test accuracy was 0.889 for gender, 0.871 for region, and 0.845 for occupation. In particular, occupation and region equalized-odds gaps increased despite lower demographic-parity gaps.

The proxy-feature ablation reduced held-out accuracy from 0.884 to 0.874 and AUC from 0.920 to 0.914. Its demographic-parity differences were 0.005 (gender), 0.099 (region), and 0.162 (occupation); equalized-odds differences were 0.097, 0.024, and 0.238, respectively. In this run, removing the features improved the gender parity gap, slightly worsened region and occupation parity gaps, and traded performance for uneven fairness changes.

## Streamlit demo

- **Applicant:** behavioural sliders, score gauge, suggested tier, local feature explanation, and bounded “How to improve” recourse.
- **Lender view:** a filterable sample of synthetic test applicants with score, suggested tier, and top local signal.
- **Fairness:** group-level approval/error-rate charts before and after mitigation, plus an opt-in proxy-feature ablation.
- **Privacy and consent:** sidebar notice, session consent checkbox, and no persistence of applicant slider inputs. Demographic controls are audit-only.

The recourse engine changes only utility on-time rate, on-time streak, recharge regularity, and savings-like balance. It never changes gender, region, or occupation. Suggested actions are model estimates, not guaranteed eligibility.

## Screenshots to capture (exactly four)

1. **Applicant score and explanation:** the score gauge, suggested loan tier, and SHAP feature chart.
2. **Recourse plan:** an applicant below the next tier with the “How to improve” message visible.
3. **Lender view:** the sample table with a score-band filter applied.
4. **Fairness audit:** group charts before/after mitigation and the proxy-feature ablation results.

Place the captures at `screenshots/01-applicant.png`, `screenshots/02-recourse.png`, `screenshots/03-lender.png`, and `screenshots/04-fairness.png` before submission.

## Deploy to a public GitHub repository and Streamlit Community Cloud

1. Sign in to GitHub and create a **public, empty** repository named `sakh` (do not initialize it with a README or license).
2. From this project folder, initialize and push the project. On the new repository page, copy its HTTPS clone URL to the clipboard, then run:

   ```powershell
   git init -b main
   git add app.py src data models README.md requirements.txt .gitignore
   git commit -m "Build Sakh synthetic credit scoring demo"
   git remote add origin (Get-Clipboard -Raw)
   git push -u origin main
   ```

   Keep the synthetic CSV and JSON metrics. The generated joblib model is intentionally ignored so deployment trains it with its pinned Python/library versions; never commit real applicant information or secrets.
3. At [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, choose **Create app**, select the public `sakh` repository and `main` branch, and set **Main file path** to `app.py`.
4. In **Advanced settings**, choose **Python 3.11**, then deploy. Streamlit installs the pinned libraries from the root `requirements.txt`.
5. Open the public app URL and verify all three tabs. If model artifacts are missing, the app trains them from the synthetic CSV on first launch.

**Common deployment fixes:** choose Python 3.11 to match the pins; verify `requirements.txt` is at the repository root; set the main file to exactly `app.py`; keep `src/__init__.py` in the pushed repository; and inspect Streamlit Cloud logs for package-install or memory errors. Do not upload secrets or real financial data.

## Responsible-AI notes and limitations

- The dataset is wholly synthetic and contains designed demographic and proxy disparities. No real person, financial account, or credit bureau record is used.
- Demographic attributes are excluded from model features, but behavioural variables (especially UPI inflow and recharge amount) may act as proxies for access, income, geography, or gender.
- Fairness metrics depend on synthetic labels, group sizes, thresholds, and the chosen metric. A lower gap is not proof of fairness; ExponentiatedGradient is shown as an experimental comparison, not production mitigation.
- SHAP describes this fitted model's associations; it does not prove causation. Recourse suggestions are simplified, may not be feasible for everyone, and do not promise score increases in the real world.
- The probability, score bands, and rupee amounts are illustrative. No affordability assessment, identity checks, consent workflow suitable for a lender, security review, legal review, or external validation is implemented.
- Before any real use: obtain meaningful informed consent; minimize and protect data; assess disparate impact with representative, lawfully collected data; involve affected communities; validate stability and calibration; establish human review, appeals, monitoring, and regulatory/legal safeguards; and consult qualified experts.

## AI tools used

Claude (Anthropic) was used for code assistance and planning as part of the project. Copilot SDK in VS Code also assisted with the implementation and validation completed in this session. Generated code and model outputs were reviewed and tested; the data and reported metrics are synthetic.

## Submission quick answers

- **Built with tags:** Python, scikit-learn, XGBoost, SHAP, Fairlearn, Streamlit, Plotly, synthetic data, explainable AI, responsible AI.
- **Project tagline (under 200 characters):** A clearer path to credit, built from everyday financial habits—not a formal credit file.

## Devpost Project Story

### Inspiration

I wanted to explore a practical version of the question behind HI-04: what if a person has a steady work life and pays everyday bills reliably, but has never had a formal credit file? I called the prototype Sakh because the word can mean credit, reputation, and standing. I focused on everyday behaviours people can recognize rather than making the score a black box.

### What it does

Sakh turns synthetic utility, mobile, UPI, income-stability, work-tenure, savings-like, and informal-loan signals into a 300–900 estimate. It shows a suggested tier, explains influential features, and gives a bounded improvement plan. A separate fairness view compares approval and error rates across gender, region, and occupation and demonstrates what happens when two proxy-like features are removed.

### How I built it

I used Python, pandas, scikit-learn, XGBoost, SHAP, Fairlearn, Plotly, and Streamlit. The deterministic generator creates 5,000 synthetic applicants and intentionally includes demographic gaps so the audit has something to measure. I compare calibrated logistic regression with calibrated XGBoost on a held-out set. The Streamlit app has applicant, lender, and fairness views; demographic columns never enter the scoring model.

### Challenges I ran into

The biggest design challenge was not treating “alternative data” as automatically fair. UPI inflow can reflect digital access and income opportunities as much as repayment habits, so excluding gender and region alone does not remove proxy effects. I kept an explicit proxy-feature ablation and made the synthetic-data and non-lending disclaimers visible. I also constrained recourse to actions a person could plausibly take, rather than suggesting they change identity or occupation.

### Accomplishments

I built the full path from deterministic data generation through model comparison, calibration, explanations, recourse, fairness metrics, and a live demo. The app can rebuild its model artifacts when they are missing, which makes the demo easier to deploy. Most importantly, each score comes with a reason and a clear statement about what the prototype cannot establish.

### What I learned

I learned that a model can omit protected columns and still reproduce unequal outcomes through its inputs and labels. Calibration, a single accuracy number, and an explanation chart each answer different questions; none makes a lending model responsible by itself. Synthetic data is useful for showing a workflow, but it cannot validate an actual credit policy.

### What's next

With more time, I would review the feature choices with gig workers and farmers, add uncertainty estimates and subgroup calibration checks, assess recourse feasibility and potential harms, and test with carefully governed representative data only after appropriate consent and legal review. Any real lending use would need independent validation, human oversight, an appeal path, and ongoing monitoring.

## 60–90 second demo video script

**Target length: about 80 seconds.** Record the app in a browser; avoid showing real personal information.

| Time | What to show / click | What to say |
|---|---|---|
| 0–10s | Open the Sakh title and sidebar disclaimer. | “Hi, I’m Paridhi Jain from Vanta. Sakh is my Neural Storm HI-04 prototype for alternative credit scoring. Everything in this demo is synthetic.” |
| 10–20s | Click **Applicant**, check the consent box, show the sliders. | “Instead of a credit bureau file, this demo uses bill-payment, recharge, UPI, income-stability, work-tenure, and savings-like signals. Gender, region, and occupation are not model inputs.” |
| 20–37s | Adjust a bill or savings slider and show the score gauge and loan tier. | “The score is a calibrated model estimate mapped to an illustrative 300-to-900 scale. The rupee tier is only a prototype suggestion, not an offer or approval.” |
| 37–50s | Point at the explanation bar chart and plain-language reasons. | “Here the app makes the strongest local signals visible. These are associations learned from synthetic labels, not causal claims.” |
| 50–61s | Show **How to improve** for a low-scoring profile. | “For a lower band, Sakh searches only realistic behavioural changes—like keeping bills on time or building a small buffer. It never recommends changing someone’s identity.” |
| 61–72s | Click **Lender view**, apply a band filter. | “The lender view is a sample of held-out synthetic applicants, with each row showing the score band and its top signal.” |
| 72–85s | Click **Fairness**, show group charts and toggle proxy-feature ablation. | “Finally, I compare group selection and error rates before and after Fairlearn mitigation, then test dropping two proxy-like features. These checks reveal trade-offs; they do not prove the system is fair.” |
