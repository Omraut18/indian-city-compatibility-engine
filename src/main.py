import pandas as pd
from recommender import calculate_score


# Load processed city data
df = pd.read_csv("data/features_processed.csv")

results = []

for _, city in df.iterrows():

    score, contributions = calculate_score(city)

    # Strongest matches
    strongest = sorted(
        contributions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    # Weakest matches
    weakest = sorted(
        contributions.items(),
        key=lambda x: x[1]
    )[:3]

    results.append({
        "city": city["city"],
        "compatibility": score,
        "strongest": strongest,
        "weakest": weakest
    })


# Convert to DataFrame
results = pd.DataFrame(results)

# Sort by compatibility
results = results.sort_values(
    "compatibility",
    ascending=False
)


# -----------------------------
# Display recommendations
# -----------------------------

print("\n========================================")
print("     AI CITY COMPATIBILITY ENGINE")
print("========================================\n")

print("Your top compatible cities:\n")


for _, row in results.head(5).iterrows():

    print("----------------------------------------")

    print(
        f"{row['city']}: "
        f"{row['compatibility']:.2f}/10"
    )

    print("\nStrongest matches:")

    for feature, contribution in row["strongest"]:
        print(f"  + {feature}")

    print("\nPotential drawbacks:")

    for feature, contribution in row["weakest"]:
        print(f"  - {feature}")

    print()


# -----------------------------
# Best match summary
# -----------------------------

best = results.iloc[0]

print("========================================")
print("           TOP RECOMMENDATION")
print("========================================")

print(
    f"\n{best['city']} "
    f"→ {best['compatibility']:.2f}/10"
)

print("\nWhy it matches you:")

for feature, contribution in best["strongest"]:
    print(f"  ✓ {feature}")

print("\nThings to consider:")

for feature, contribution in best["weakest"]:
    print(f"  ⚠ {feature}")

print()