# Revolut Feature Cannibalization Analysis

> A product analytics study exploring how feature adoption 
> patterns affect revenue mix in a super-app — inspired by 
> Revolut's multi-product model.

 **[Live Dashboard →](https://revolut-analysis-wexh4vuywg5r8lgpmqlaff.streamlit.app/)**

---

## The Business Problem

Revolut operates as a super-app with 10+ financial products 
in one interface. When users adopt one feature heavily, do they 
disengage from others? This study investigates whether rising 
**crypto usage cannibalizes savings vault behavior** — and what 
that means for revenue planning and product strategy.

---

## Key Findings

-  **Cannibalization confirmed** — Crypto usage shows a negative lagged correlation of r = -0.207 with savings vault usage at a 2-month delay
-  **48.6% of users are At-Risk** — concentrated crypto behavior with vulnerable savings engagement
-  **£3.7M annualised revenue at risk** — based on simplified per-feature revenue assumptions
-  **India shows 4.6x higher international transfer usage** vs dataset average — strong regional behavioral signal

---

## Methodology

| Layer | Tools | Purpose |
|---|---|---|
| Data Generation | Python, NumPy | Synthetic 50K users, 24 months, realistic correlations |
| SQL Analysis | SQLite | Adoption rates, trends, segmentation, lag insight |
| Python Analysis | Pandas, Scikit-learn | K-Means clustering, lagged cross-correlation, revenue simulation |
| Dashboard | Streamlit, Plotly | Interactive 6-page product intelligence tool |

---

## Project Structure

```
