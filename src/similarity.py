import pandas as pd
import numpy as np

from preferences import get_preferences
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
# 5. Create user vector
# -----------------------------

preferences = get_preferences()

user_vector = np.array([
    preferences[feature]
    for feature in features
]).reshape(1, -1)


# -----------------------------
# 6. Standardize user vector
# -----------------------------

user_vector_scaled = scaler.transform(user_vector)


# -----------------------------
# 7. Calculate similarity
# -----------------------------

similarities = cosine_similarity(
    user_vector_scaled,
    city_matrix_scaled
)[0]


# -----------------------------
# 8. Store results
# -----------------------------

results = pd.DataFrame({
    "city": df["city"],
    "similarity": similarities
})


# -----------------------------
# 9. Sort
# -----------------------------

results = results.sort_values(
    "similarity",
    ascending=False
)


# -----------------------------
# 10. Display
# -----------------------------

print("\n========================================")
print("     STANDARDIZED SIMILARITY")
print("========================================\n")

for _, row in results.iterrows():

    print(
        f"{row['city']}: "
        f"{row['similarity']:.4f}"
    )