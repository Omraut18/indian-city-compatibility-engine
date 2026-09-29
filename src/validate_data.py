import pandas as pd


# -----------------------------
# 1. Load raw data
# -----------------------------

df = pd.read_csv("data/raw_features.csv")


# -----------------------------
# 2. Check missing values
# -----------------------------

print("\n========================================")
print("          DATA VALIDATION")
print("========================================\n")

print("Missing values:\n")

print(df.isnull().sum())


# -----------------------------
# 3. Check duplicate cities
# -----------------------------

duplicates = df["city"].duplicated().sum()

print("\nDuplicate cities:", duplicates)


# -----------------------------
# 4. Check score ranges
# -----------------------------

score_columns = [
    "cafes_est",
    "fitness_est",
    "nightlife_est",
    "social_est"
]

print("\nChecking lifestyle scores...")

for column in score_columns:

    invalid = df[
        (df[column] < 1) |
        (df[column] > 10)
    ]

    print(
        f"{column}: "
        f"{len(invalid)} invalid values"
    )


# -----------------------------
# 5. Final result
# -----------------------------

print("\nValidation complete.")