import pandas as pd

raw = pd.read_csv("data/affordability_raw.csv")

raw["affordability"] = (
    10 - (
        (raw["cost_plus_rent_index"] - raw["cost_plus_rent_index"].min())
        / (raw["cost_plus_rent_index"].max() - raw["cost_plus_rent_index"].min())
    ) * 9
).round(2)

cities = pd.read_csv("data/cities.csv")

cities = cities.drop(columns=["affordability"], errors="ignore")

cities = cities.merge(
    raw[["city", "affordability"]],
    on="city",
    how="left"
)

cities.to_csv("data/cities_processed.csv", index=False)

print(cities)