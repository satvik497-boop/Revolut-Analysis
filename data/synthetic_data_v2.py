import numpy as np
import pandas as pd

# ============================================================
# PARAMETERS
# ============================================================

N_USERS  = 50000
N_MONTHS = 24
SEED     = 42

np.random.seed(SEED)

# ============================================================
# USER-LEVEL ATTRIBUTES
# ============================================================

user_ids = np.arange(1, N_USERS + 1)

# Premium flag (25% of users)
premium = np.random.binomial(1, 0.25, N_USERS)

# Age: 18-45, younger users skew toward crypto
age = np.random.normal(loc=29, scale=6, size=N_USERS).clip(18, 55).astype(int)

# Country: Revolut's top markets with realistic distribution
countries     = ["UK", "India", "Germany", "France", "Poland"]
country_probs = [0.30,  0.25,    0.20,      0.15,     0.10]
country       = np.random.choice(countries, size=N_USERS, p=country_probs)

# Latent engagement score (lognormal = realistic heavy tail)
engagement = np.random.lognormal(mean=0.0, sigma=0.6, size=N_USERS)

# Premium users are 30% more active
engagement *= np.where(premium == 1, 1.30, 1.00)

users = pd.DataFrame({
    "user_id":    user_ids,
    "premium":    premium,
    "age":        age,
    "country":    country,
    "engagement": engagement
})

# ============================================================
# CREATE PANEL (user x month)
# ============================================================

months = np.arange(1, N_MONTHS + 1)

df = (
    users.assign(key=1)
         .merge(pd.DataFrame({"month": months, "key": 1}), on="key")
         .drop("key", axis=1)
)

# ============================================================
# SEASONALITY
# ============================================================

df["seasonality"] = (
    1
    + 0.10 * np.sin(2 * np.pi * df["month"] / 12)
    + 0.05 * np.cos(2 * np.pi * df["month"] / 6)
)

# ============================================================
# COUNTRY-LEVEL MULTIPLIERS
# Realistic behavioral differences by market
# ============================================================

country_multipliers = {
    # (card, bank_transfer, intl_transfer, stock, crypto, savings)
    "UK":      (1.20, 1.00, 0.80, 1.30, 1.10, 1.20),
    "India":   (0.90, 1.10, 1.80, 0.80, 1.20, 0.70),  # high intl transfers
    "Germany": (1.00, 1.20, 0.70, 1.10, 0.80, 1.30),  # high savings
    "France":  (1.10, 0.90, 0.90, 0.90, 0.90, 1.00),
    "Poland":  (0.85, 1.00, 1.20, 0.70, 1.00, 0.90),
}

cm = pd.DataFrame(
    country_multipliers,
    index=["m_card","m_bank","m_intl","m_stock","m_crypto","m_savings"]
).T.reset_index().rename(columns={"index":"country"})

df = df.merge(cm, on="country", how="left")

# ============================================================
# AGE-BASED MULTIPLIERS
# Younger users (18-25) skew toward crypto, older toward savings
# ============================================================

df["age_crypto_boost"]  = np.where(df["age"] <= 25, 1.40, 1.00)
df["age_savings_boost"] = np.where(df["age"] >= 35, 1.30, 1.00)

# ============================================================
# BASE PRODUCT USAGE
# ============================================================

noise = lambda n, s: np.random.normal(0, s, n)
n = len(df)

df["card_payments"] = (
    20 * df["engagement"] * df["seasonality"] * df["m_card"]
    + noise(n, 5)
)

df["bank_transfers"] = (
    5 * df["engagement"] * df["seasonality"] * df["m_bank"]
    + noise(n, 2)
)

df["intl_transfers"] = (
    2 * df["engagement"] * df["seasonality"] * df["m_intl"]
    + noise(n, 1)
)

df["stock_trading"] = (
    1.8 * df["engagement"] * df["m_stock"]
    + noise(n, 1.2)
)

df["crypto_usage"] = (
    2.5 * df["engagement"] * df["m_crypto"] * df["age_crypto_boost"]
    + 0.8 * df["stock_trading"]
    + noise(n, 1.5)
)

df["savings_vaults"] = (
    4 * df["engagement"] * df["seasonality"] * df["m_savings"] * df["age_savings_boost"]
    + noise(n, 1.5)
)

# ============================================================
# CLIP NEGATIVES
# ============================================================

product_cols = [
    "card_payments", "bank_transfers", "intl_transfers",
    "stock_trading", "crypto_usage", "savings_vaults"
]

for col in product_cols:
    df[col] = np.clip(df[col], 0, None)

# ============================================================
# 2-MONTH LAG EFFECT: HIGH CRYPTO -> LOWER SAVINGS
# ============================================================

df = df.sort_values(["user_id", "month"])

df["crypto_lag_2"] = (
    df.groupby("user_id")["crypto_usage"]
      .shift(2)
      .fillna(0)
)

# Cannibalization: crypto 2 months ago reduces savings today
df["savings_vaults"] = (
    4 * df["engagement"] * df["seasonality"] * df["m_savings"] * df["age_savings_boost"]
    - 1.2 * df["crypto_lag_2"]
    + np.random.normal(0, 1.2, len(df))
)

df["savings_vaults"] = np.clip(df["savings_vaults"], 0, None)

# ============================================================
# ROUND TO REALISTIC INTEGER COUNTS
# ============================================================

for col in product_cols:
    df[col] = np.round(df[col]).astype(int)

# ============================================================
# VERIFY KEY CORRELATIONS
# ============================================================

lag_corr = df["crypto_lag_2"].corr(df["savings_vaults"])
print(f"Crypto(t-2) vs Savings(t) correlation: {lag_corr:.3f}")
print("Expected: negative (around -0.3 to -0.5) ✓" if lag_corr < -0.2 else "⚠ Check lag logic")

# ============================================================
# FINAL DATASET
# ============================================================

final_cols = [
    "user_id", "month", "premium", "age", "country",
    "card_payments", "bank_transfers", "savings_vaults",
    "crypto_usage", "stock_trading", "intl_transfers"
]

dataset = df[final_cols]

print(f"\nDataset shape: {dataset.shape}")
print(dataset.head())

dataset.to_csv("revolut_usage_50k_users_v2.csv", index=False)
print("\n Saved: revolut_usage_50k_users_v2.csv")
