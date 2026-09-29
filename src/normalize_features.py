import pandas as pd


# -----------------------------
# 1. Load raw feature data
# -----------------------------

df = pd.read_csv("data/raw_features.csv")


# -----------------------------
# 2. Add affordability
# -----------------------------

affordability = pd.read_csv("data/affordability_raw.csv")

affordability["affordability"] = (
    10 - (
        (affordability["cost_plus_rent_index"]
         - affordability["cost_plus_rent_index"].min())
        /
        (
            affordability["cost_plus_rent_index"].max()
            - affordability["cost_plus_rent_index"].min()
        )
    ) * 9
).round(2)

df = df.merge(
    affordability[["city", "affordability"]],
    on="city",
    how="left"
)


# -----------------------------
# 3. Convert raw values
# -----------------------------

raw_features = [
    "career_raw",
    "mobility_raw",
    "safety_raw",
    "nature_raw"
]

for feature in raw_features:
    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )


# -----------------------------
# 4. Normalize raw features
# -----------------------------

for feature in raw_features:

    score_name = feature.replace("_raw", "")

    min_value = df[feature].min()
    max_value = df[feature].max()

    df[score_name] = (
        1 + (
            (df[feature] - min_value)
            / (max_value - min_value)
        ) * 9
    ).round(1)


# -----------------------------
# 5. Business ranking
# Lower rank = better
# -----------------------------

df["business_raw"] = pd.to_numeric(
    df["business_raw"],
    errors="coerce"
)

df["business"] = (
    10 - (
        (df["business_raw"] - df["business_raw"].min())
        /
        (
            df["business_raw"].max()
            - df["business_raw"].min()
        )
    ) * 9
).round(1)


# -----------------------------
# 6. Climate score
# 24°C is our target comfort temperature
# -----------------------------

df["climate_avg_temp"] = pd.to_numeric(
    df["climate_avg_temp"],
    errors="coerce"
)

df["climate"] = (
    10 - abs(df["climate_avg_temp"] - 24) * 0.5
).clip(1, 10).round(1)


# -----------------------------
# 7. Lifestyle estimates
# -----------------------------

estimated_features = [
    "cafes_est",
    "fitness_est",
    "nightlife_est",
    "social_est"
]

for feature in estimated_features:

    score_name = feature.replace("_est", "")

    df[score_name] = df[feature]


# -----------------------------
# 8. Save processed dataset
# -----------------------------

df.to_csv(
    "data/features_processed.csv",
    index=False
)


# -----------------------------
# 9. Display final features
# -----------------------------

print("\nProcessed city features:\n")

print(
    df[
        [
            "city",
            "affordability",
            "career",
            "business",
            "cafes",
            "fitness",
            "nightlife",
            "nature",
            "climate",
            "mobility",
            "safety",
            "social"
        ]
    ]
)