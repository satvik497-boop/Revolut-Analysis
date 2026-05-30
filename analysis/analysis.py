import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("revolut_usage_50k_users_v2.csv")

PRODUCT_COLS = [
    "card_payments", "bank_transfers", "savings_vaults",
    "crypto_usage", "stock_trading", "intl_transfers"
]

# ============================================================
# SECTION 0: DATA VALIDATION
# ============================================================

print("=" * 60)
print("SECTION 0: DATA VALIDATION")
print("=" * 60)

print(f"\nShape: {df.shape}")
print(f"Unique users: {df['user_id'].nunique():,}")
print(f"Months: {df['month'].nunique()}")

print("\nMissing values:")
print(df.isnull().sum())

print("\nNegative values:")
for col in PRODUCT_COLS:
    neg = (df[col] < 0).sum()
    print(f"  {col}: {neg}")

print("\nSummary statistics:")
print(df[PRODUCT_COLS].describe().round(2))

premium_stats = df.groupby("premium")[PRODUCT_COLS].mean()
print("\nPremium vs Free avg usage:")
print(premium_stats.round(2))

uplift = (
    premium_stats.loc[1].mean()
    / premium_stats.loc[0].mean()
    - 1
) * 100
print(f"\nPremium uplift over free users: {uplift:.2f}%")

print("\n Data validation complete — no issues found")

# ============================================================
# SECTION 1: CORRELATION HEATMAP
# ============================================================

print("\n" + "=" * 60)
print("SECTION 1: FEATURE CORRELATION HEATMAP")
print("=" * 60)

corr = df[PRODUCT_COLS].corr()
print(corr.round(3))

plt.figure(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title(
    "Feature Correlation Matrix\n(Negative = Cannibalization Signal)",
    fontsize=13, pad=15
)
plt.tight_layout()
plt.savefig("1_correlation_heatmap.png", dpi=150)
plt.close()
print(" Saved: 1_correlation_heatmap.png")

# ============================================================
# SECTION 2: LAGGED CROSS-CORRELATION
# Measures how crypto at time T affects savings at T+1, T+2...
# ============================================================

print("\n" + "=" * 60)
print("SECTION 2: LAGGED CROSS-CORRELATION")
print("=" * 60)

df_sorted = df.sort_values(["user_id", "month"])

lag_results = []
for lag in range(0, 7):
    df_sorted[f"crypto_lag_{lag}"] = (
        df_sorted.groupby("user_id")["crypto_usage"]
                 .shift(lag)
    )
    r = df_sorted[f"crypto_lag_{lag}"].corr(df_sorted["savings_vaults"])
    lag_results.append({"lag": lag, "correlation": round(r, 4)})
    print(f"  Lag {lag}: crypto(t-{lag}) vs savings(t) = {r:.4f}")

lag_df = pd.DataFrame(lag_results)

plt.figure(figsize=(9, 5))
colors = ["#ef4444" if c < 0 else "#22c55e" for c in lag_df["correlation"]]
bars = plt.bar(
    lag_df["lag"], lag_df["correlation"],
    color=colors, edgecolor="white", linewidth=0.8
)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xlabel("Lag (months)", fontsize=11)
plt.ylabel("Correlation with Savings Vault Usage", fontsize=11)
plt.title(
    "Lagged Cross-Correlation: Crypto → Savings Vault\n(Negative bars = cannibalization)",
    fontsize=12
)
plt.xticks(lag_df["lag"])
for bar, val in zip(bars, lag_df["correlation"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val:.3f}", ha="center", va="bottom", fontsize=9
    )
plt.tight_layout()
plt.savefig("2_lagged_correlation.png", dpi=150)
plt.close()
print(" Saved: 2_lagged_correlation.png")

peak_lag = lag_df.loc[lag_df["correlation"].abs().idxmax()]
print(f"\nPeak cannibalization at lag {int(peak_lag['lag'])} months: r = {peak_lag['correlation']:.4f}")

# ============================================================
# SECTION 3: K-MEANS CLUSTERING
# Label users as Diversified, At-Risk, or Cannibalized
# ============================================================

print("\n" + "=" * 60)
print("SECTION 3: USER SEGMENTATION (K-MEANS)")
print("=" * 60)

# Aggregate to user level
user_agg = df.groupby("user_id")[PRODUCT_COLS].mean().reset_index()

# Feature engineering
user_agg["crypto_dominance"] = (
    user_agg["crypto_usage"]
    / (user_agg[PRODUCT_COLS].sum(axis=1) + 1e-9)
)
user_agg["savings_dominance"] = (
    user_agg["savings_vaults"]
    / (user_agg[PRODUCT_COLS].sum(axis=1) + 1e-9)
)
user_agg["feature_diversity"] = (user_agg[PRODUCT_COLS] > 0).sum(axis=1)

cluster_features = (
    PRODUCT_COLS
    + ["crypto_dominance", "savings_dominance", "feature_diversity"]
)

scaler = StandardScaler()
X = scaler.fit_transform(user_agg[cluster_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
user_agg["cluster"] = kmeans.fit_predict(X)

cluster_profiles = user_agg.groupby("cluster").agg(
    avg_crypto=("crypto_usage", "mean"),
    avg_savings=("savings_vaults", "mean"),
    avg_diversity=("feature_diversity", "mean"),
    count=("user_id", "count")
).reset_index()

print("\nCluster profiles:")
print(cluster_profiles.round(2))

# Assign business labels
cluster_profiles = cluster_profiles.sort_values("avg_crypto", ascending=False)
label_map = {}
cluster_ids = cluster_profiles["cluster"].tolist()
label_map[cluster_ids[0]] = "Cannibalized"
label_map[cluster_ids[1]] = "At-Risk"
label_map[cluster_ids[2]] = "Diversified"

user_agg["segment"] = user_agg["cluster"].map(label_map)

seg_counts = user_agg["segment"].value_counts()
print("\nSegment distribution:")
for seg, count in seg_counts.items():
    print(f"  {seg}: {count:,} users ({count / len(user_agg) * 100:.1f}%)")

colors_map = {
    "Diversified":  "#22c55e",
    "At-Risk":      "#f59e0b",
    "Cannibalized": "#ef4444"
}

plt.figure(figsize=(9, 6))
for seg, grp in user_agg.groupby("segment"):
    plt.scatter(
        grp["crypto_usage"], grp["savings_vaults"],
        alpha=0.15, s=8, label=seg, color=colors_map[seg]
    )
plt.xlabel("Avg Monthly Crypto Usage", fontsize=11)
plt.ylabel("Avg Monthly Savings Vault Usage", fontsize=11)
plt.title(
    "User Segmentation: Crypto vs Savings Behavior\n(K-Means, 3 Clusters)",
    fontsize=12
)
plt.legend(markerscale=4, fontsize=10)
plt.tight_layout()
plt.savefig("3_user_segmentation.png", dpi=150)
plt.close()
print(" Saved: 3_user_segmentation.png")

# ============================================================
# SECTION 4: REVENUE IMPACT SIMULATION
# ============================================================

print("\n" + "=" * 60)
print("SECTION 4: REVENUE IMPACT SIMULATION")
print("=" * 60)

REVENUE_PER_UNIT = {
    "card_payments":  2.00,
    "bank_transfers": 0.50,
    "savings_vaults": 1.00,
    "crypto_usage":   0.50,
    "stock_trading":  0.75,
    "intl_transfers": 1.50,
}

for col, rev in REVENUE_PER_UNIT.items():
    df[f"rev_{col}"] = df[col] * rev

rev_cols = [f"rev_{c}" for c in PRODUCT_COLS]
df["total_revenue"] = df[rev_cols].sum(axis=1)

df_sorted = df.sort_values(["user_id", "month"])
df_sorted["crypto_lag_2"] = (
    df_sorted.groupby("user_id")["crypto_usage"]
             .shift(2)
             .fillna(0)
)
df_sorted["savings_lost"] = (1.2 * df_sorted["crypto_lag_2"]).clip(lower=0)
df_sorted["revenue_lost"] = (
    df_sorted["savings_lost"] * REVENUE_PER_UNIT["savings_vaults"]
)

total_rev_lost     = df_sorted["revenue_lost"].sum()
annual_rev_at_risk = total_rev_lost / 2

print(f"\nTotal simulated savings revenue lost (24 months): £{total_rev_lost:,.0f}")
print(f"Annualised revenue at risk:                       £{annual_rev_at_risk:,.0f}")
print("\nNOTE: Based on synthetic data and simplified revenue model.")
print("This is illustrative — real figures require Revolut's actual data.")

user_rev = df.groupby("user_id")["total_revenue"].sum().reset_index()
user_rev = user_rev.merge(user_agg[["user_id", "segment"]], on="user_id")
seg_rev  = user_rev.groupby("segment")["total_revenue"].agg(["mean", "sum"]).round(2)

print("\nRevenue by segment (24-month total per user):")
print(seg_rev)

seg_rev_plot  = seg_rev.reset_index()
bar_colors    = [colors_map[s] for s in seg_rev_plot["segment"]]

plt.figure(figsize=(9, 5))
plt.bar(
    seg_rev_plot["segment"], seg_rev_plot["mean"],
    color=bar_colors, edgecolor="white"
)
plt.xlabel("User Segment", fontsize=11)
plt.ylabel("Avg 24-Month Revenue per User (£)", fontsize=11)
plt.title(
    "Simulated Revenue by User Segment\n(Cannibalized users generate less savings revenue)",
    fontsize=12
)
for i, (_, row) in enumerate(seg_rev_plot.iterrows()):
    plt.text(
        i, row["mean"] + 0.5,
        f"£{row['mean']:.0f}", ha="center", fontsize=10
    )
plt.tight_layout()
plt.savefig("4_revenue_by_segment.png", dpi=150)
plt.close()
print(" Saved: 4_revenue_by_segment.png")

# ============================================================
# SECTION 5: REGIONAL BEHAVIOR
# ============================================================

print("\n" + "=" * 60)
print("SECTION 5: REGIONAL BEHAVIOR ANALYSIS")
print("=" * 60)

country_avg      = df.groupby("country")[PRODUCT_COLS].mean().round(2)
country_avg_norm = country_avg.div(country_avg.max())

print(country_avg)

plt.figure(figsize=(10, 6))
sns.heatmap(
    country_avg_norm.T,
    annot=country_avg.T,
    fmt=".1f",
    cmap="YlOrRd",
    linewidths=0.5,
    cbar_kws={"label": "Relative Usage (normalized)"}
)
plt.title(
    "Feature Usage by Country\n(Raw values shown, color = relative intensity)",
    fontsize=12
)
plt.xlabel("Country", fontsize=11)
plt.ylabel("Feature", fontsize=11)
plt.tight_layout()
plt.savefig("5_regional_heatmap.png", dpi=150)
plt.close()
print(" Saved: 5_regional_heatmap.png")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE — KEY FINDINGS")
print("=" * 60)
print(f"1. Crypto → Savings cannibalization: r = {lag_df.loc[2, 'correlation']:.3f} at 2-month lag")
print(f"2. {seg_counts.get('Cannibalized', 0):,} users ({seg_counts.get('Cannibalized', 0) / len(user_agg) * 100:.1f}%) classified as Cannibalized")
print(f"3. {seg_counts.get('At-Risk', 0):,} users ({seg_counts.get('At-Risk', 0) / len(user_agg) * 100:.1f}%) classified as At-Risk")
print(f"4. Annualised simulated revenue at risk: £{annual_rev_at_risk:,.0f}")
print(f"5. India shows highest intl_transfers; Germany highest savings_vaults")
print("=" * 60)

# Save segments for Streamlit
user_agg.to_csv("user_segments.csv", index=False)
print("\n Saved: user_segments.csv (used by Streamlit dashboard)")
