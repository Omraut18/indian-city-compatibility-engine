import pandas as pd
import numpy as np

from preferences import get_preferences
from recommender import calculate_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 1. Load city data
# -----------------------------

df = pd.read_csv("data/features_processed.csv")


# -----------------------------
# 2. Features
# -----------------------------

features = [
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


# -----------------------------
# 3. Create city matrix
# -----------------------------

city_matrix = df[features].values


# -----------------------------
# 4. Standardize city data
# -----------------------------

scaler = StandardScaler()

city_matrix_scaled = scaler.fit_transform(city_matrix)


# -----------------------------
# 5. User preference vector
# -----------------------------

preferences = get_preferences()

user_vector = np.array([
    preferences[feature]
    for feature in features
]).reshape(1, -1)


# Standardize user vector
user_vector_scaled = scaler.transform(user_vector)


# -----------------------------
# 6. Calculate results
# -----------------------------

results = []

for i, (_, city) in enumerate(df.iterrows()):

    # Weighted compatibility
    compatibility, _ = calculate_score(
    city,
    preferences
)

    # Cosine similarity
    similarity = cosine_similarity(
        user_vector_scaled,
        city_matrix_scaled[i].reshape(1, -1)
    )[0][0]

    # Convert similarity from -1..1
    # into approximately 0..10
    similarity_score = (
        (similarity + 1) / 2
    ) * 10

    # Hybrid score
    hybrid_score = (
    0.8 * compatibility
    + 0.2 * similarity_score
)

    results.append({
        "city": city["city"],
        "compatibility": compatibility,
        "similarity": similarity_score,
        "hybrid": hybrid_score
    })


# -----------------------------
# 7. Sort results
# -----------------------------

results = pd.DataFrame(results)

results = results.sort_values(
    "hybrid",
    ascending=False
)


# -----------------------------
# 8. Display
# -----------------------------

print("\n========================================")
print("       HYBRID CITY RECOMMENDER")
print("========================================\n")

for _, row in results.iterrows():

    print(
        f"{row['city']}: "
        f"{row['hybrid']:.2f}/10 "
        f"(Compatibility: {row['compatibility']:.2f} | "
        f"Similarity: {row['similarity']:.2f})"
    )