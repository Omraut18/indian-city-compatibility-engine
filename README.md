# 🏙️ Indian City Compatibility Engine

### Find Indian cities that match the lifestyle you want.

🔴 **Live Demo:** https://indian-city-compatibility-engin.streamlit.app/

---

## 📌 About the Project

The Indian City Compatibility Engine is a data-driven recommendation system that helps users find Indian cities based on their personal lifestyle priorities.

Instead of simply ranking cities by population, GDP, or cost of living, the system asks:

> **"Based on the life I want, which Indian city should I actually live in?"**

Users rate the importance of 11 lifestyle factors from 1–10, and the engine calculates how well each city matches those preferences.

---

## ⚙️ How It Works

The system uses a hybrid recommendation approach.

### 1. User Preferences

Users rate:

- 💰 Affordability
- 💼 Career opportunities
- 🚀 Business / Startups
- ☕ Cafes & lifestyle
- 🏋️ Fitness
- 🌃 Nightlife
- 🌿 Nature
- 🌤️ Climate
- 🚇 Transportation
- 🛡️ Safety
- 🧑‍🤝‍🧑 Social life

### 2. Compatibility Score

Each city is compared against the user's preferences.

The system gives greater influence to factors that the user considers more important.

### 3. Cosine Similarity

The system also uses cosine similarity to measure how similar the user's overall preference profile is to each city's feature profile.

### 4. Hybrid Score

The final score combines both approaches:

**Hybrid Score = 80% Compatibility + 20% Preference Similarity**

Cities are then ranked according to their final compatibility score.

---

## 🧠 Machine Learning / Data Concepts

This project uses:

- Weighted scoring
- Feature normalization
- StandardScaler
- Cosine similarity
- Hybrid recommendation
- Data validation
- Feature engineering
- Interactive data visualization

---

## 🛠️ Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Streamlit**
- **Plotly**
- **Git & GitHub**

---

## 🗂️ Project Structure

```text
City compatibility engine/
│
├── data/
│   ├── raw_features.csv
│   ├── affordability_raw.csv
│   └── features_processed.csv
│
├── src/
│   ├── app.py
│   ├── hybrid.py
│   ├── recommender.py
│   ├── preferences.py
│   ├── normalize_features.py
│   ├── validate_data.py
│   ├── run_engine.py
│   └── similarity.py
│
├── .gitignore
├── README.md
└── requirements.txt