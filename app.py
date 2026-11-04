"""Sakh: synthetic, explainable alternative-credit scoring demo."""
from __future__ import annotations

import json
import os
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_gen import DEMOGRAPHIC_COLS, FEATURE_COLS
from src.fairness import proxy_feature_comparison, run_fairness_audit
from src.model import (FAIRNESS_PATH, SCORE_BANDS, explain_applicant,
                       load_or_train, predict_applicant)
from src.recourse import recommend_recourse

ROOT = os.path.dirname(os.path.abspath(__file__))
FEATURE_LABELS = {
    "bill_on_time_rate": "On-time utility bills",
    "bill_streak_months": "Longest bill streak",
    "recharge_regularity": "Mobile recharge regularity",
    "avg_recharge_amount": "Average recharge amount",
    "upi_inflow_avg": "Average monthly UPI inflow",
    "upi_inflow_cv": "UPI inflow variation",
    "income_volatility": "Income volatility",
    "seasonal_swing": "Seasonal income swing",
    "active_months": "Active work months",
    "min_balance_ratio": "Savings-like balance",
    "informal_loan_burden": "Informal loan burden",
}
FEATURE_NOTES = {
    "bill_on_time_rate": "on-time utility payments",
    "bill_streak_months": "a longer bill-payment streak",
    "recharge_regularity": "regular mobile recharges",
    "avg_recharge_amount": "the average recharge amount",
    "upi_inflow_avg": "average UPI inflows",
    "upi_inflow_cv": "variation in UPI inflows",
    "income_volatility": "income volatility",
    "seasonal_swing": "seasonal income swings",
    "active_months": "time active at work",
    "min_balance_ratio": "a savings-like balance buffer",
    "informal_loan_burden": "existing informal-loan burden",
}

st.set_page_config(page_title="Sakh | Alternative credit", page_icon="साख", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:wght@400;500;600;700&family=Tiro+Devanagari+Hindi&display=swap');
    :root {
      --paper: #F3EAD3; --card: #FBF6E8; --rule: #D8CBA8;
      --ink: #1E2A5A; --vermilion: #9E2A2B; --brass: #B8862B;
      --green: #3E6B4E;
    }
    html, body, [data-testid="stAppViewContainer"] {
      color: var(--ink);
      font-family: "Source Serif 4", Georgia, serif;
    }
    .stApp {
      color: var(--ink);
      background-color: var(--paper);
      background-image: repeating-linear-gradient(
        to bottom, rgba(216, 203, 168, .30) 0, rgba(216, 203, 168, .30) 1px,
        transparent 1px, transparent 32px
      );
    }
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] [data-testid="stToolbar"] {
      background: transparent;
    }
    .stAppDeployButton { display: none; }
    h1, h2, h3, h4, h5, h6 {
      color: var(--ink) !important;
      font-family: "Fraunces", Georgia, serif !important;
      letter-spacing: -.02em;
    }
    p, label, li, span, div, small {
      color: var(--ink);
    }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
      color: var(--ink) !important;
    }
    [data-testid="stAppViewContainer"] .main .block-container {
      max-width: 1440px;
      padding-top: 1.6rem;
      padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
      background: var(--vermilion);
      border-right: 1px solid rgba(243, 234, 211, .35);
      overflow-x: hidden;
    }
    [data-testid="stSidebarContent"] { overflow-x: hidden; }
    [data-testid="stSidebar"] * {
      color: var(--paper) !important;
      font-family: "Source Serif 4", Georgia, serif;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
      color: var(--paper) !important;
    }
    [data-testid="stSidebar"] input { accent-color: var(--brass); }
    [data-testid="stSidebar"] [data-testid="stCheckbox"] p {
      color: var(--paper) !important;
    }
    .sidebar-notice {
      border: 1px solid rgba(243, 234, 211, .72);
      border-left: 4px solid var(--brass);
      border-radius: 3px;
      padding: .8rem .9rem;
      margin: .75rem 0;
      color: var(--paper) !important;
      background: rgba(30, 42, 90, .13);
      font-size: .92rem;
      line-height: 1.45;
    }
    .sidebar-notice * { color: var(--paper) !important; }
    [data-testid="stTabs"] [role="tablist"] {
      gap: .45rem;
      border-bottom: 1px solid var(--rule);
      flex-wrap: wrap;
    }
    [data-testid="stTabs"] [role="tab"] {
      color: var(--ink) !important;
      background: rgba(251, 246, 232, .72);
      border: 1px solid var(--rule);
      border-bottom: 3px solid transparent;
      border-radius: 4px 4px 0 0;
      padding: .55rem 1rem;
      font-family: "Fraunces", Georgia, serif;
    }
    [data-testid="stTabs"] [role="tab"] *,
    [data-testid="stTabs"] [role="tab"] p {
      color: var(--ink) !important;
      font-family: "Fraunces", Georgia, serif !important;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
      color: var(--ink) !important;
      background: var(--card);
      border-bottom: 3px solid var(--vermilion);
    }
    [data-testid="stSlider"] label, [data-testid="stSlider"] label *,
    [data-testid="stMultiSelect"] label, [data-testid="stSelectbox"] label,
    [data-testid="stToggle"] label, [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] label *, [data-testid="stWidgetLabel"] * {
      color: var(--ink) !important;
    }
    [data-testid="stSlider"] [data-rac][style*="left"] {
      background-color: var(--vermilion) !important;
      border-color: var(--vermilion) !important;
      box-shadow: 0 0 0 2px var(--card);
    }
    [data-testid="stSlider"] [data-testid="stSliderTickBar"],
    [data-testid="stSlider"] [data-testid="stSliderTickBar"] * {
      color: var(--ink) !important;
    }
    [data-baseweb="select"] > div, [data-baseweb="popover"] > div,
    [data-baseweb="menu"] {
      color: var(--ink) !important;
      background: var(--card) !important;
      border-color: var(--rule) !important;
    }
    [role="option"], [role="option"] * {
      color: var(--ink) !important;
      background: var(--card) !important;
    }
    [data-testid="stMetric"] {
      background: var(--card);
      border: 1px solid var(--rule);
      border-left: 5px solid var(--brass);
      border-radius: 3px;
      padding: 1rem 1.15rem;
      min-height: 112px;
      box-shadow: 2px 3px 0 rgba(30, 42, 90, .04);
    }
    [data-testid="stMetric"] label, [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
      color: var(--ink) !important;
      font-variant: small-caps;
      letter-spacing: .06em;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
      color: var(--ink) !important;
      font-family: "IBM Plex Mono", monospace !important;
      font-size: clamp(1.25rem, 2.2vw, 2rem) !important;
      font-weight: 600;
      overflow-wrap: anywhere;
    }
    [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {
      color: var(--ink) !important;
    }
    .sakh-masthead { margin: .2rem 0 1.15rem; }
    .sakh-title {
      color: var(--ink);
      font-family: "Fraunces", Georgia, serif;
      font-size: clamp(2.2rem, 5vw, 3.5rem);
      font-weight: 700;
      line-height: 1.05;
      letter-spacing: -.035em;
    }
    .sakh-title .devanagari {
      font-family: "Tiro Devanagari Hindi", serif;
      font-weight: 400;
    }
    .sakh-tagline {
      color: var(--ink);
      font-family: "Source Serif 4", Georgia, serif;
      font-size: clamp(1rem, 2vw, 1.2rem);
      margin-top: .55rem;
    }
    .sakh-stamp {
      display: inline-block;
      margin: .85rem 0 .25rem;
      padding: .28rem .55rem;
      border: 1px solid var(--vermilion);
      color: var(--vermilion) !important;
      background: rgba(251, 246, 232, .8);
      font: 600 .68rem "IBM Plex Mono", monospace;
      letter-spacing: .08em;
      transform: rotate(-3deg);
    }
    .sakh-note { color: var(--ink); font-size: .98rem; }
    .ledger-card, .ledger-alert {
      color: var(--ink) !important;
      background: var(--card);
      border: 1px solid var(--rule);
      border-left: 5px solid var(--brass);
      border-radius: 3px;
      padding: 1rem 1.15rem;
      margin: .65rem 0;
      line-height: 1.55;
      box-shadow: 2px 3px 0 rgba(30, 42, 90, .04);
    }
    .ledger-alert { border-left-color: var(--vermilion); }
    .ledger-card *, .ledger-alert * { color: var(--ink) !important; }
    .ledger-label {
      color: var(--vermilion) !important;
      font: 600 .72rem "IBM Plex Mono", monospace;
      letter-spacing: .09em;
      text-transform: uppercase;
    }
    .ledger-score {
      color: var(--ink) !important;
      font: 600 1.35rem "IBM Plex Mono", monospace;
      white-space: normal;
    }
    .ledger-actions { margin: .45rem 0 0; padding-left: 1.2rem; }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
      background: var(--card);
      border: 1px solid var(--rule);
    }
    [data-testid="stDataFrame"] * { color: var(--ink); }
    button, [data-testid="stBaseButton-secondary"] {
      color: var(--paper) !important;
      background: var(--vermilion) !important;
      border-color: var(--vermilion) !important;
    }
    button p, button span { color: var(--paper) !important; }
    @media (max-width: 700px) {
      [data-testid="stAppViewContainer"] .main .block-container {
        padding: 1rem .85rem 2rem;
      }
      [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap;
        gap: .5rem;
      }
      [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        min-width: min(100%, 260px);
        flex: 1 1 100%;
      }
      [data-testid="stTabs"] [role="tab"] { padding: .45rem .65rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cached_bundle():
    return load_or_train()


@st.cache_data(show_spinner="Scoring the synthetic sample...")
def portfolio_rows(_bundle):
    sample = _bundle["test_df"].head(30).copy()
    scored = []
    for _, row in sample.iterrows():
        applicant = {feature: float(row[feature]) for feature in FEATURE_COLS}
        result = predict_applicant(_bundle, applicant)
        _, impacts = explain_applicant(_bundle, applicant)
        feature, impact = impacts[0]
        reason = FEATURE_NOTES[feature]
        if impact < 0:
            reason = reason[0].upper() + reason[1:] + " lowered the estimate"
        else:
            reason = reason[0].upper() + reason[1:] + " supported the estimate"
        scored.append({
            "Applicant": row.get("applicant_id", "Synthetic"),
            "Occupation": row.get("occupation", ""),
            "Score": result["score"],
            "Band": result["band"],
            "Suggested loan": f"₹{result['loan_inr']:,}" if result["loan_inr"] else "Not yet eligible",
            "Decision signal": reason,
        })
    return pd.DataFrame(scored)


@st.cache_data(show_spinner="Re-training without proxy-like features...")
def cached_proxy_comparison(_bundle):
    return proxy_feature_comparison(_bundle)


def load_fairness(bundle):
    if os.path.exists(FAIRNESS_PATH):
        with open(FAIRNESS_PATH, encoding="utf-8") as file:
            return json.load(file)
    return run_fairness_audit(bundle)


def reasons_text(feature, impact):
    direction = "raised" if impact >= 0 else "lowered"
    return f"{FEATURE_NOTES[feature].capitalize()} {direction} this estimate."


def applicant_tab(bundle, consent):
    st.subheader("Applicant snapshot")
    st.caption("Use synthetic or consented, self-reported behavioural values. Demographics are audit-only.")
    if not consent:
        st.markdown(
            '<div class="ledger-alert">Please confirm the consent notice in the sidebar '
            'before calculating an applicant score.</div>',
            unsafe_allow_html=True,
        )
        return

    left, right = st.columns(2)
    values = {}
    with left:
        values["bill_on_time_rate"] = st.slider("Utility bills paid on time (last 12 months)", 0.25, 1.0, 0.76, 0.01)
        values["bill_streak_months"] = st.slider("Longest on-time bill streak (months)", 0, 12, 7)
        values["recharge_regularity"] = st.slider("Mobile recharge regularity", 0.2, 1.0, 0.74, 0.01)
        values["avg_recharge_amount"] = st.slider("Average mobile recharge (₹)", 49, 799, 180, 5)
        values["upi_inflow_avg"] = st.slider("Average monthly UPI inflow (₹)", 1000, 60000, 14000, 500)
        values["upi_inflow_cv"] = st.slider("UPI inflow variation (coefficient of variation)", 0.08, 1.5, 0.42, 0.01)
    with right:
        values["income_volatility"] = st.slider("Income volatility", 0.02, 1.0, 0.30, 0.01)
        values["seasonal_swing"] = st.slider("Seasonal income swing", 0.05, 2.5, 0.45, 0.01)
        values["active_months"] = st.slider("Months active in work", 3, 120, 24)
        values["min_balance_ratio"] = st.slider("Months maintaining a small balance buffer", 0.0, 1.0, 0.42, 0.01)
        values["informal_loan_burden"] = st.slider("Informal loan repayments / monthly income", 0.0, 0.8, 0.12, 0.01)
    result = predict_applicant(bundle, values)
    score_col, loan_col, band_col = st.columns(3)
    score_col.metric("Sakh score", f"{result['score']} / 900")
    loan_col.metric("Suggested maximum", f"₹{result['loan_inr']:,}" if result["loan_inr"] else "Not yet eligible")
    band_col.metric("Score band", result["band"])
    st.caption(
        f"Estimated repayment probability: {result['probability']:.1%}. "
        "This probability is calibrated on synthetic labels; it is not a real-world risk estimate."
    )
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=result["score"],
        number={"suffix": " / 900"},
        gauge={
            "axis": {"range": [300, 900]},
            "bar": {"color": "#B8862B"},
            "steps": [
                {"range": [300, 600], "color": "rgba(158, 42, 43, .23)"},
                {"range": [600, 700], "color": "rgba(184, 134, 43, .34)"},
                {"range": [700, 800], "color": "rgba(62, 107, 78, .25)"},
                {"range": [800, 900], "color": "rgba(62, 107, 78, .42)"},
            ],
        },
    ))
    gauge.update_layout(
        height=250, margin={"t": 25, "b": 5, "l": 25, "r": 25},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FBF6E8",
        font={"color": "#1E2A5A", "family": "Source Serif 4, Georgia, serif"},
    )
    gauge.update_traces(
        number={"font": {"color": "#1E2A5A", "family": "IBM Plex Mono, monospace"}},
        gauge={"bar": {"color": "#B8862B"}, "bordercolor": "#D8CBA8"},
    )
    st.plotly_chart(gauge, use_container_width=True)

    method, impacts = explain_applicant(bundle, values)
    st.markdown(f"#### Why this score? ({method})")
    reasons = impacts[:6]
    chart = pd.DataFrame({
        "Feature": [FEATURE_LABELS[name] for name, _ in reasons],
        "Impact on repayment estimate": [float(value) for _, value in reasons],
    }).sort_values("Impact on repayment estimate")
    chart["Direction"] = np.where(chart["Impact on repayment estimate"] >= 0, "Raised", "Lowered")
    figure = px.bar(
        chart, x="Impact on repayment estimate", y="Feature", color="Direction",
        color_discrete_map={"Raised": "#3E6B4E", "Lowered": "#9E2A2B"},
        orientation="h",
    )
    figure.update_layout(
        height=330, showlegend=False, margin={"l": 8, "r": 8, "t": 15, "b": 15},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FBF6E8",
        font={"color": "#1E2A5A", "family": "Source Serif 4, Georgia, serif"},
        xaxis={"gridcolor": "#D8CBA8", "zerolinecolor": "#1E2A5A"},
        yaxis={"gridcolor": "#D8CBA8"},
    )
    st.plotly_chart(figure, use_container_width=True)
    for feature, impact in reasons[:4]:
        st.write("• " + reasons_text(feature, float(impact)))

    st.markdown("#### How to improve")
    plan = recommend_recourse(bundle, values)
    action_items = "".join(
        f"<li>{escape(action)}</li>" for action in plan.get("actions", [])
    )
    if plan.get("target_score") is not None:
        score_line = (
            f'<div class="ledger-score">{plan["before"]["score"]} '
            f'&rarr; {plan["after"]["score"]}</div>'
        )
        message = (
            f'<div class="ledger-label">Your next entry:</div>{score_line}'
            f'<ul class="ledger-actions">{action_items}</ul>'
            f'<div>{escape(plan["message"])}</div>'
        )
    else:
        message = f'<div class="ledger-label">Your next entry:</div><div>{escape(plan["message"])}</div>'
    st.markdown(
        f'<div class="ledger-card">{message}</div>',
        unsafe_allow_html=True,
    )


def lender_tab(bundle):
    st.subheader("Lender view")
    st.caption("A small sample from the held-out synthetic set; not real applicants or lending recommendations.")
    rows = portfolio_rows(bundle)
    band_filter = st.multiselect(
        "Show score bands", options=[band["name"] for band in SCORE_BANDS],
        default=[band["name"] for band in SCORE_BANDS],
    )
    visible = rows[rows["Band"].isin(band_filter)]
    st.dataframe(visible, hide_index=True, use_container_width=True)


def fairness_tab(bundle, fairness):
    st.subheader("Fairness & model checks")
    st.caption(
        "Protected attributes are excluded from model inputs. Metrics below are measured on "
        "the same held-out synthetic sample; mitigation is an experiment, not a guarantee."
    )
    attribute = st.selectbox("Audit groups", DEMOGRAPHIC_COLS, format_func=lambda value: value.title())
    result = fairness["by_attribute"][attribute]
    before = result["before"]
    after = result["after"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("DP difference", f"{before['demographic_parity_difference']:.3f}", delta=f"{after['demographic_parity_difference'] - before['demographic_parity_difference']:+.3f} after mitigation")
    m2.metric("Equalized-odds difference", f"{before['equalized_odds_difference']:.3f}", delta=f"{after['equalized_odds_difference'] - before['equalized_odds_difference']:+.3f} after mitigation")
    m3.metric("Before accuracy", f"{before['accuracy']:.1%}")
    m4.metric("After accuracy", f"{after['accuracy']:.1%}")

    groups = sorted(set(before["by_group"]) | set(after["by_group"]))
    charts = []
    for metric, label in [
        ("selection_rate", "Approval / selection rate"),
        ("true_positive_rate", "True positive rate"),
        ("false_positive_rate", "False positive rate"),
    ]:
        for stage, source in [("Before mitigation", before), ("After mitigation", after)]:
            for group in groups:
                charts.append({
                    "Group": group,
                    "Rate": source["by_group"].get(group, {}).get(metric, 0.0),
                    "Stage": stage,
                    "Metric": label,
                })
    fig = px.bar(
        pd.DataFrame(charts), x="Group", y="Rate", color="Stage",
        barmode="group", facet_row="Metric",
        color_discrete_map={"Before mitigation": "#1E2A5A", "After mitigation": "#B8862B"},
        range_y=[0, 1],
    )
    fig.update_layout(
        height=680, margin={"t": 30, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FBF6E8",
        font={"color": "#1E2A5A", "family": "Source Serif 4, Georgia, serif"},
    )
    fig.update_xaxes(gridcolor="#D8CBA8", linecolor="#D8CBA8")
    fig.update_yaxes(gridcolor="#D8CBA8", linecolor="#D8CBA8")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Mitigation: {fairness['method']}. Lower gaps are not automatically fair outcomes.")

    drop_proxy = st.toggle("Drop proxy-like features and retrain", value=False)
    if drop_proxy:
        comparison = cached_proxy_comparison(bundle)
        st.markdown("#### Proxy feature ablation")
        st.write("Dropped: " + ", ".join(comparison["dropped_features"]))
        comparison_rows = [
            {"Model": "All features", **bundle["metrics"][f"{bundle['model_name']} (calibrated)"]},
            {"Model": "Proxy features dropped", **comparison["performance"]},
        ]
        st.dataframe(pd.DataFrame(comparison_rows).set_index("Model").round(3), use_container_width=True)
        proxy_group = comparison["by_attribute"][attribute]
        st.write(
            f"{attribute.title()} after removing proxy-like features: "
            f"demographic parity difference "
            f"{proxy_group['demographic_parity_difference']:.3f}; "
            f"equalized-odds difference {proxy_group['equalized_odds_difference']:.3f}."
        )


def main():
    st.markdown(
        '<div class="sakh-masthead">'
        '<div class="sakh-title"><span class="devanagari">साख</span> · Sakh</div>'
        '<div class="sakh-tagline">Credit standing, written in the language of everyday work.</div>'
        '<div class="sakh-stamp">SYNTHETIC DATA · PROTOTYPE</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.title("About Sakh")
    st.sidebar.write(
        "Sakh (साख) means credit, reputation, and standing. This hackathon prototype "
        "explores credit scoring for gig workers and farmers with thin or no formal credit files."
    )
    consent = st.sidebar.checkbox(
        "I consent to process the values I enter for this session. They are not saved by this demo.",
        value=False,
    )
    st.sidebar.markdown(
        '<div class="sidebar-notice"><strong>Privacy</strong><br>'
        'Do not enter real personal identifiers or account data. Demographics are audit-only; '
        'predictions use behavioural features. This prototype does not store applicant inputs.</div>'
        '<div class="sidebar-notice"><strong>All data is synthetic.</strong><br>'
        'This is a decision-support prototype, not a real lending decision.</div>',
        unsafe_allow_html=True,
    )

    bundle = cached_bundle()
    fairness = load_fairness(bundle)
    tabs = st.tabs(["Applicant", "Lender view", "Fairness"])
    with tabs[0]:
        applicant_tab(bundle, consent)
    with tabs[1]:
        lender_tab(bundle)
    with tabs[2]:
        fairness_tab(bundle, fairness)


if __name__ == "__main__":
    main()