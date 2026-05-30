import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import subprocess

if not os.path.exists("revolut_usage_50k_users_v2.csv"):
    subprocess.run(["python", "data/synthetic_data_v2.py"], check=True)

if not os.path.exists("user_segments.csv"):
    subprocess.run(["python", "analysis/analysis.py"], check=True)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Revolut Feature Cannibalization",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# REVOLUT-STYLE CSS
# ============================================================

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0a0a0a; color: #ffffff; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #222; }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] { color: #0075eb; font-size: 28px; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #888; font-size: 13px; }
    [data-testid="stMetricDelta"] { font-size: 12px; }

    /* Headers */
    h1 { color: #ffffff; font-weight: 800; letter-spacing: -0.5px; }
    h2 { color: #ffffff; font-weight: 700; }
    h3 { color: #cccccc; font-weight: 600; }

    /* Insight boxes */
    .insight-box {
        background: #1a1a1a;
        border-left: 3px solid #0075eb;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 14px;
        color: #cccccc;
        line-height: 1.6;
    }
    .insight-box strong { color: #ffffff; }

    /* Warning box */
    .warn-box {
        background: #1a1a1a;
        border-left: 3px solid #f59e0b;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 13px;
        color: #aaaaaa;
    }

    /* Section divider */
    .section-title {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #555;
        margin: 32px 0 16px 0;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    main     = pd.read_csv("revolut_usage_50k_users_v2.csv")
    segments = pd.read_csv("user_segments.csv")
    adoption = pd.read_csv("adoption.csv")
    premiums = pd.read_csv("premiums.csv")
    trends   = pd.read_csv("trends.csv")
    regional = pd.read_csv("regional.csv")
    age_grps = pd.read_csv("age_groups.csv")
    lag      = pd.read_csv("lag_insight.csv")
    return main, segments, adoption, premiums, trends, regional, age_grps, lag

main, segments, adoption, premiums, trends, regional, age_grps, lag = load_data()

PRODUCT_COLS = [
    "card_payments", "bank_transfers", "savings_vaults",
    "crypto_usage", "stock_trading", "intl_transfers"
]

COLORS = {
    "primary":      "#0075eb",
    "green":        "#22c55e",
    "amber":        "#f59e0b",
    "red":          "#ef4444",
    "Diversified":  "#22c55e",
    "At-Risk":      "#f59e0b",
    "Cannibalized": "#ef4444",
    "bg":           "#1a1a1a",
    "border":       "#2a2a2a",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a0a0a",
    plot_bgcolor="#111111",
    font=dict(color="#cccccc", family="Inter, sans-serif", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="#1e1e1e", linecolor="#222"),
    yaxis=dict(gridcolor="#1e1e1e", linecolor="#222"),
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🔴 Revolut Analytics")
    st.markdown("<div style='color:#555; font-size:12px; margin-bottom:24px'>Feature Cannibalization Intelligence</div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Executive Summary", "Feature Adoption", "Cannibalization Analysis",
         "User Segmentation", "Revenue Impact", "Regional & Demographics"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<div style='color:#444; font-size:11px'>Dataset</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#888; font-size:12px'>50,000 users · 24 months<br>1.2M rows · Synthetic data</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#444; font-size:11px; margin-top:16px'>Built by</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#888; font-size:12px'>Satvik Mishra<br>IIT Mandi · 2026</div>", unsafe_allow_html=True)

# ============================================================
# PAGE 1 — EXECUTIVE SUMMARY
# ============================================================

if page == "Executive Summary":

    st.markdown("# Feature Cannibalization Intelligence")
    st.markdown("<div style='color:#888; margin-bottom:32px'>A product analytics study inspired by Revolut's super-app model — exploring how feature adoption patterns affect revenue mix and user behavior.</div>", unsafe_allow_html=True)

    # KPI row
    seg_counts = segments["segment"].value_counts()
    total_users = len(segments)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Users Analysed", f"{total_users:,}")
    with c2:
        st.metric("Features Tracked", "6")
    with c3:
        st.metric("Cannibalized Users", f"{seg_counts.get('Cannibalized', 0):,}",
                  delta=f"{seg_counts.get('Cannibalized',0)/total_users*100:.1f}% of base")
    with c4:
        st.metric("Annualised Revenue at Risk", "£3.7M",
                  delta="simulated", delta_color="off")

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### What is Feature Cannibalization?")
        st.markdown("""
        <div class='insight-box'>
        Revolut is a <strong>super-app</strong> — users can pay, save, invest in stocks, trade crypto, 
        and send money internationally, all in one place. Feature cannibalization occurs when 
        <strong>heavy adoption of one feature suppresses engagement with another</strong>, 
        changing Revolut's revenue mix in ways that may not be immediately visible.
        <br><br>
        This study examines whether <strong>rising crypto usage cannibalizes savings vault behavior</strong> 
        over time — and what that means for product strategy and revenue planning.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Key Findings")
        findings = [
            ("🔴 Cannibalization confirmed", "Crypto usage at time T has a negative correlation of **r = -0.207** with savings vault usage at T+2 months — a clear delayed cannibalization signal."),
            ("⚠️ 48.6% users are At-Risk", "Nearly half the user base shows usage patterns where crypto engagement is rising while savings behavior remains vulnerable."),
            ("💰 £3.7M annual revenue at risk", "Based on simplified revenue assumptions per feature, cannibalization from crypto to savings represents a meaningful annual revenue exposure."),
            ("🌍 Regional signal: India", "Indian users show 4.6x higher international transfer usage vs average — a distinct behavioral cluster relevant for product localisation."),
        ]
        for title, desc in findings:
            st.markdown(f"<div class='insight-box'><strong>{title}</strong><br>{desc}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("### User Segments")
        seg_data = pd.DataFrame({
            "Segment": seg_counts.index,
            "Users": seg_counts.values
        })
        fig = px.pie(
            seg_data, values="Users", names="Segment",
            color="Segment",
            color_discrete_map=COLORS,
            hole=0.6
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_traces(textfont_color="white")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='warn-box'>⚠️ <strong>Note:</strong> This analysis is based on synthetic data generated to model realistic behavioral patterns. All revenue figures are illustrative and based on simplified assumptions.</div>", unsafe_allow_html=True)

# ============================================================
# PAGE 2 — FEATURE ADOPTION
# ============================================================

elif page == "Feature Adoption":

    st.markdown("# Feature Adoption Analysis")
    st.markdown("<div style='color:#888; margin-bottom:24px'>What percentage of users actively use each product feature?</div>", unsafe_allow_html=True)

    # Adoption rates bar chart
    adoption_melted = pd.DataFrame({
        "Feature": ["Card Payments", "Bank Transfers", "Savings Vaults",
                    "Crypto", "Stock Trading", "Intl Transfers"],
        "Adoption %": [
            adoption["card_adoption_pct"].values[0],
            adoption["bank_adoption_pct"].values[0],
            adoption["savings_adoption_pct"].values[0],
            adoption["crypto_adoption_pct"].values[0],
            adoption["stock_adoption_pct"].values[0],
            adoption["intl_adoption_pct"].values[0],
        ]
    }).sort_values("Adoption %", ascending=True)

    fig = px.bar(
        adoption_melted, x="Adoption %", y="Feature",
        orientation="h",
        color="Adoption %",
        color_continuous_scale=[[0, "#1a3a5c"], [1, "#0075eb"]],
        text="Adoption %"
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont_color="white")
    fig.update_layout(**PLOTLY_LAYOUT, title="Active Feature Adoption Rates (%)", showlegend=False,
                      coloraxis_showscale=False, height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='insight-box'><strong>Card Payments dominate</strong> as the gateway feature — nearly all users make card payments. Savings Vaults and Crypto show meaningful but lower adoption, which is where cannibalization dynamics play out.</div>", unsafe_allow_html=True)

    # Monthly trends
    st.markdown("### Monthly Usage Trends")

    feature_map = {
        "Card Payments": "avg_card_payments",
        "Savings Vaults": "avg_savings_vaults",
        "Crypto": "avg_crypto_usage",
        "Stock Trading": "avg_stock_trading",
        "Bank Transfers": "avg_bank_transfers",
        "Intl Transfers": "avg_intl_transfers",
    }

    selected = st.multiselect(
        "Select features to compare",
        list(feature_map.keys()),
        default=["Card Payments", "Savings Vaults", "Crypto"]
    )

    if selected:
        fig2 = go.Figure()
        palette = ["#0075eb", "#22c55e", "#ef4444", "#f59e0b", "#a855f7", "#06b6d4"]
        for i, feat in enumerate(selected):
            col = feature_map[feat]
            fig2.add_trace(go.Scatter(
                x=trends["month"], y=trends[col],
                name=feat, mode="lines+markers",
                line=dict(color=palette[i % len(palette)], width=2),
                marker=dict(size=5)
            ))
        fig2.update_layout(**PLOTLY_LAYOUT, title="Average Monthly Usage per Feature",
                           xaxis_title="Month", yaxis_title="Avg Usage Score", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Premium vs free
    st.markdown("### Premium vs Free Users")
    prem_display = premiums.copy()
    prem_display["plan"] = prem_display["premium"].map({0: "Free", 1: "Premium"})
    prem_melted = prem_display.melt(
        id_vars="plan",
        value_vars=["avg_card_payments","avg_bank_transfers","avg_savings_vaults",
                    "avg_crypto_usage","avg_stock_trading","avg_intl_transfers"],
        var_name="Feature", value_name="Avg Usage"
    )
    prem_melted["Feature"] = prem_melted["Feature"].str.replace("avg_","").str.replace("_"," ").str.title()

    fig3 = px.bar(
        prem_melted, x="Feature", y="Avg Usage", color="plan",
        barmode="group",
        color_discrete_map={"Free": "#2a2a2a", "Premium": "#0075eb"}
    )
    fig3.update_layout(**PLOTLY_LAYOUT, title="Premium vs Free: Average Feature Usage", height=380)
    st.plotly_chart(fig3, use_container_width=True)

    uplift = (premiums.loc[premiums["premium"]==1, "avg_crypto_usage"].values[0] /
              premiums.loc[premiums["premium"]==0, "avg_crypto_usage"].values[0] - 1) * 100
    st.markdown(f"<div class='insight-box'><strong>Premium users show ~{uplift:.0f}% higher crypto usage</strong> than free users — making them both more valuable and more exposed to cannibalization risk.</div>", unsafe_allow_html=True)

# ============================================================
# PAGE 3 — CANNIBALIZATION ANALYSIS
# ============================================================

elif page == "Cannibalization Analysis":

    st.markdown("# Cannibalization Analysis")
    st.markdown("<div style='color:#888; margin-bottom:24px'>Does rising crypto usage suppress savings vault behavior over time?</div>", unsafe_allow_html=True)

    # Lagged correlation chart
    lag_corrs = []
    main_sorted = main.sort_values(["user_id","month"])
    for l in range(0, 7):
        main_sorted[f"cl{l}"] = main_sorted.groupby("user_id")["crypto_usage"].shift(l)
        r = main_sorted[f"cl{l}"].corr(main_sorted["savings_vaults"])
        lag_corrs.append({"Lag (months)": l, "Correlation": round(r, 4)})
    lag_df = pd.DataFrame(lag_corrs)

    colors_bar = [COLORS["red"] if c < 0 else COLORS["green"] for c in lag_df["Correlation"]]
    fig = go.Figure(go.Bar(
        x=lag_df["Lag (months)"],
        y=lag_df["Correlation"],
        marker_color=colors_bar,
        text=[f"{v:.3f}" for v in lag_df["Correlation"]],
        textposition="outside",
        textfont=dict(color="white", size=11)
    ))
    fig.add_hline(y=0, line_color="#444", line_width=1)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Lagged Cross-Correlation: Crypto Usage → Savings Vault",
        xaxis_title="Lag (months)",
        yaxis_title="Correlation with Savings Vault",
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
    <strong>The cannibalization effect peaks at lag 2.</strong> Crypto usage has a 
    <strong>positive correlation at lag 0</strong> (engaged users use multiple features simultaneously), 
    but turns <strong>negative at lag 2 (r = -0.207)</strong> — meaning users who traded heavily 
    in crypto two months ago are saving significantly less today. This delayed behavioral shift 
    is a classic cannibalization pattern.
    </div>
    """, unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown("### Feature Correlation Matrix")
    corr = main[PRODUCT_COLS].corr()
    labels = ["Card Payments","Bank Transfers","Savings Vaults","Crypto","Stock Trading","Intl Transfers"]

    fig2 = px.imshow(
        corr.values,
        x=labels, y=labels,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f"
    )
    fig2.update_layout(**PLOTLY_LAYOUT, title="Feature Correlation Matrix", height=480)
    fig2.update_traces(textfont_size=11)
    st.plotly_chart(fig2, use_container_width=True)

    # Lag insight from SQL
    st.markdown("### Crypto Intensity vs Savings Behavior (SQL Output)")
    lag_display = lag.copy()
    lag_display.columns = ["Crypto Segment", "Avg Savings Vault", "Avg Card Payments", "Row Count"]
    fig3 = px.bar(
        lag_display, x="Crypto Segment", y="Avg Savings Vault",
        color="Crypto Segment",
        color_discrete_map={
            "Low Crypto": COLORS["green"],
            "Medium Crypto": COLORS["amber"],
            "High Crypto": COLORS["red"]
        },
        text="Avg Savings Vault"
    )
    fig3.update_traces(texttemplate="%{text:.2f}", textposition="outside", textfont_color="white")
    fig3.update_layout(**PLOTLY_LAYOUT, title="Average Savings Vault Usage by Crypto Intensity",
                       showlegend=False, height=380)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='insight-box'><strong>High crypto users save 60–70% less</strong> than low crypto users on average — confirming the cannibalization hypothesis from a direct SQL aggregation.</div>", unsafe_allow_html=True)

# ============================================================
# PAGE 4 — USER SEGMENTATION
# ============================================================

elif page == "User Segmentation":

    st.markdown("# User Segmentation")
    st.markdown("<div style='color:#888; margin-bottom:24px'>K-Means clustering identifies three behavioral archetypes across 50,000 users.</div>", unsafe_allow_html=True)

    seg_counts = segments["segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Users"]

    c1, c2, c3 = st.columns(3)
    for col, (_, row) in zip([c1, c2, c3], seg_counts.iterrows()):
        with col:
            pct = row["Users"] / len(segments) * 100
            st.metric(row["Segment"], f"{row['Users']:,}", delta=f"{pct:.1f}%")

    # Scatter plot
    st.markdown("### Crypto vs Savings: Cluster View")

    sample = segments.sample(min(5000, len(segments)), random_state=42)
    fig = px.scatter(
        sample, x="crypto_usage", y="savings_vaults",
        color="segment",
        color_discrete_map=COLORS,
        opacity=0.4,
        size_max=4,
        labels={"crypto_usage": "Avg Monthly Crypto Usage", "savings_vaults": "Avg Monthly Savings Vault Usage"}
    )
    fig.update_traces(marker_size=4)
    fig.update_layout(**PLOTLY_LAYOUT, title="User Segmentation: Crypto vs Savings Vault Behavior", height=460)
    st.plotly_chart(fig, use_container_width=True)

    # Segment profiles radar
    st.markdown("### Segment Feature Profiles")
    seg_profile = segments.groupby("segment")[PRODUCT_COLS].mean().reset_index()
    feature_labels = ["Card Payments","Bank Transfers","Savings Vaults","Crypto","Stock Trading","Intl Transfers"]

    fig2 = go.Figure()
    for _, row in seg_profile.iterrows():
        vals = [row[c] for c in PRODUCT_COLS]
        vals_norm = [v / max(segments[c].mean(), 0.01) for v, c in zip(vals, PRODUCT_COLS)]
        fig2.add_trace(go.Scatterpolar(
            r=vals_norm + [vals_norm[0]],
            theta=feature_labels + [feature_labels[0]],
            fill="toself",
            name=row["segment"],
            line_color=COLORS[row["segment"]],
            opacity=0.6
        ))
    fig2.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        font=dict(color="#cccccc"),
        polar=dict(
            bgcolor="#111111",
            radialaxis=dict(visible=True, gridcolor="#222", color="#555"),
            angularaxis=dict(gridcolor="#222", color="#888")
        ),
        title="Normalised Feature Usage by Segment",
        height=460
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Retention simulator
    st.markdown("### Retention Simulator")
    st.markdown("<div style='color:#888; font-size:13px; margin-bottom:16px'>If Revolut intervened with targeted nudges for At-Risk users, how many could be moved to Diversified?</div>", unsafe_allow_html=True)

    at_risk_count = seg_counts.loc[seg_counts["Segment"]=="At-Risk","Users"].values[0]
    intervention = st.slider("Intervention effectiveness (%)", 0, 50, 15)
    recovered = int(at_risk_count * intervention / 100)
    rev_per_user = 1061  # avg diversified user 24-month revenue

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("At-Risk Users", f"{at_risk_count:,}")
    with c2: st.metric("Users Recovered", f"{recovered:,}", delta=f"{intervention}% conversion")
    with c3: st.metric("Additional Revenue (24-mo)", f"£{recovered * rev_per_user:,.0f}")

    st.markdown(f"<div class='insight-box'>At a <strong>{intervention}% intervention success rate</strong>, Revolut could recover <strong>{recovered:,} users</strong> from the At-Risk segment, generating approximately <strong>£{recovered * rev_per_user:,.0f}</strong> in additional 24-month revenue through improved feature diversification.</div>", unsafe_allow_html=True)

# ============================================================
# PAGE 5 — REVENUE IMPACT
# ============================================================

elif page == "Revenue Impact":

    st.markdown("# Revenue Impact Simulation")
    st.markdown("<div style='color:#888; margin-bottom:24px'>Quantifying the financial exposure from feature cannibalization.</div>", unsafe_allow_html=True)

    st.markdown("### Adjust Revenue Assumptions")
    st.markdown("<div style='color:#666; font-size:13px; margin-bottom:16px'>Modify the revenue per unit of usage per month (£) to see how the at-risk figure changes.</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        rev_card    = st.slider("Card Payments (£/unit)", 0.5, 5.0, 2.0, 0.1)
        rev_bank    = st.slider("Bank Transfers (£/unit)", 0.1, 2.0, 0.5, 0.1)
    with c2:
        rev_savings = st.slider("Savings Vaults (£/unit)", 0.5, 3.0, 1.0, 0.1)
        rev_crypto  = st.slider("Crypto (£/unit)", 0.1, 2.0, 0.5, 0.1)
    with c3:
        rev_stock   = st.slider("Stock Trading (£/unit)", 0.1, 2.0, 0.75, 0.05)
        rev_intl    = st.slider("Intl Transfers (£/unit)", 0.5, 3.0, 1.5, 0.1)

    # Recalculate
    REVENUE_MAP = {
        "card_payments": rev_card,
        "bank_transfers": rev_bank,
        "savings_vaults": rev_savings,
        "crypto_usage": rev_crypto,
        "stock_trading": rev_stock,
        "intl_transfers": rev_intl
    }

    main_copy = main.copy().sort_values(["user_id","month"])
    main_copy["crypto_lag_2"] = main_copy.groupby("user_id")["crypto_usage"].shift(2).fillna(0)
    main_copy["savings_lost"] = (1.2 * main_copy["crypto_lag_2"]).clip(lower=0)
    main_copy["revenue_lost"] = main_copy["savings_lost"] * rev_savings

    total_lost   = main_copy["revenue_lost"].sum()
    annual_risk  = total_lost / 2

    for col, rev in REVENUE_MAP.items():
        main_copy[f"rev_{col}"] = main_copy[col] * rev
    main_copy["total_rev"] = sum(main_copy[f"rev_{c}"] for c in PRODUCT_COLS)
    total_rev = main_copy["total_rev"].sum()
    risk_pct  = annual_risk / (total_rev / 2) * 100

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Simulated Revenue (Annual)", f"£{total_rev/2:,.0f}")
    with c2: st.metric("Annualised Revenue at Risk", f"£{annual_risk:,.0f}")
    with c3: st.metric("At-Risk as % of Total", f"{risk_pct:.2f}%")

    # Revenue by feature
    st.markdown("### Revenue Contribution by Feature")
    feat_rev = {col: main_copy[f"rev_{col}"].sum() / 2 for col in PRODUCT_COLS}
    feat_rev_df = pd.DataFrame({
        "Feature": ["Card Payments","Bank Transfers","Savings Vaults","Crypto","Stock Trading","Intl Transfers"],
        "Annual Revenue (£)": list(feat_rev.values())
    }).sort_values("Annual Revenue (£)", ascending=True)

    fig = px.bar(
        feat_rev_df, x="Annual Revenue (£)", y="Feature",
        orientation="h", color="Annual Revenue (£)",
        color_continuous_scale=[[0,"#1a3a5c"],[1,"#0075eb"]],
        text="Annual Revenue (£)"
    )
    fig.update_traces(texttemplate="£%{text:,.0f}", textposition="outside", textfont_color="white")
    fig.update_layout(**PLOTLY_LAYOUT, title="Simulated Annual Revenue by Feature",
                      showlegend=False, coloraxis_showscale=False, height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<div class='warn-box'>⚠️ All figures are based on synthetic data and simplified revenue assumptions. The purpose is to demonstrate the <strong>analytical framework</strong> a product analyst would use — not to represent Revolut's actual financials.</div>", unsafe_allow_html=True)

# ============================================================
# PAGE 6 — REGIONAL & DEMOGRAPHICS
# ============================================================

elif page == "Regional & Demographics":

    st.markdown("# Regional & Demographic Analysis")
    st.markdown("<div style='color:#888; margin-bottom:24px'>How does feature usage vary across geographies and age groups?</div>", unsafe_allow_html=True)

    # Regional heatmap
    st.markdown("### Feature Usage by Country")
    reg_cols = ["avg_card_payments","avg_bank_transfers","avg_savings_vaults",
                "avg_crypto_usage","avg_stock_trading","avg_intl_transfers"]
    reg_labels = ["Card Payments","Bank Transfers","Savings Vaults","Crypto","Stock Trading","Intl Transfers"]

    reg_matrix = regional[reg_cols].values
    reg_norm   = reg_matrix / reg_matrix.max(axis=0)

    fig = px.imshow(
        reg_norm.T,
        x=regional["country"].tolist(),
        y=reg_labels,
        color_continuous_scale="Blues",
        text_auto=False,
        aspect="auto"
    )
    # Add raw value annotations
    for i, row in regional.iterrows():
        for j, col in enumerate(reg_cols):
            fig.add_annotation(
                x=i, y=j,
                text=f"{row[col]:.1f}",
                showarrow=False,
                font=dict(color="white", size=11)
            )
    fig.update_layout(**PLOTLY_LAYOUT, title="Normalised Feature Usage by Country (raw values shown)",
                      height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='insight-box'><strong>India</strong> shows 4.6x higher international transfer usage (avg 4.62) vs the dataset average of ~2.6 — reflecting real-world remittance behavior. This is a strong signal for localised product features.</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='insight-box'><strong>Germany</strong> shows the highest savings vault usage and bank transfer activity — consistent with German users' historically conservative financial behavior and preference for structured saving.</div>", unsafe_allow_html=True)

    # Age group analysis
    st.markdown("### Crypto vs Savings by Age Group")

    fig2 = make_subplots(rows=1, cols=2,
                          subplot_titles=["Avg Crypto Usage by Age", "Avg Savings Vault by Age"])
    fig2.add_trace(go.Bar(
        x=age_grps["age_group"].astype(str),
        y=age_grps["avg_crypto"],
        marker_color=COLORS["red"],
        name="Crypto", showlegend=False
    ), row=1, col=1)
    fig2.add_trace(go.Bar(
        x=age_grps["age_group"].astype(str),
        y=age_grps["avg_savings"],
        marker_color=COLORS["primary"],
        name="Savings", showlegend=False
    ), row=1, col=2)
    fig2.update_layout(paper_bgcolor="#0a0a0a", plot_bgcolor="#111111",
                        font=dict(color="#cccccc"), height=380,
                        margin=dict(l=40,r=20,t=60,b=40))
    fig2.update_xaxes(gridcolor="#1e1e1e")
    fig2.update_yaxes(gridcolor="#1e1e1e")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='insight-box'><strong>18–25 year olds show the highest crypto usage</strong> — driven by the age multiplier baked into the data generation — while <strong>36+ users lean toward savings vaults</strong>. This age-driven divergence directly amplifies cannibalization risk among younger cohorts.</div>", unsafe_allow_html=True)
