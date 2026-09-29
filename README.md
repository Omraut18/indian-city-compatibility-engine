# 🏙️ Indian City Compatibility Engine

An AI-inspired recommendation system that helps users find Indian cities that match their preferred lifestyle.

Instead of simply asking which city is "best", the system asks:

> **"Which Indian city best matches the life I want?"**

The user rates how important different lifestyle factors are, and the engine compares those preferences with city-level feature scores.

---

## 🎯 Project Goal

The goal is to build a personalized city recommendation system based on **lifestyle compatibility** rather than simply ranking cities by population, GDP, or popularity.

Users can prioritize factors such as:

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

The system then calculates compatibility scores for each city and produces a personalized ranking.

---

# ⚙️ How It Works

The recommendation pipeline is:

```text
User Preferences
       ↓
Preference Vector
       ↓
City Feature Data
       ↓
Feature Normalization
       ↓
Weighted Compatibility
       ↓
Cosine Similarity
       ↓
Hybrid Score
       ↓
City Recommendations