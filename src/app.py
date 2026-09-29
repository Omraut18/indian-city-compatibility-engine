import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="City Compatibility Engine",
    page_icon="🏙️",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏙️ Indian City Compatibility Engine")

st.write(
    "Find Indian cities that match the lifestyle you want."
)

st.divider()


# ============================================================
# USER PREFERENCES
# ============================================================

st.subheader("🎯 Rate Your Priorities")

st.write(
    "1 = Not important  |  10 = Extremely important"
)

affordability = st.slider(
    "💰 Affordability", 1, 10, 5
)

career = st.slider(
    "💼 Career opportunities", 1, 10, 5
)

business = st.slider(
    "🚀 Business / Startups", 1, 10, 5
)

cafes = st.slider(
    "☕ Cafes & lifestyle", 1, 10, 5
)

fitness = st.slider(
    "🏋️ Fitness", 1, 10, 5
)

nightlife = st.slider(
    "🌃 Nightlife", 1, 10, 5
)

nature = st.slider(
    "🌿 Nature", 1, 10, 5
)

climate = st.slider(
    "🌤️ Climate", 1, 10, 5
)

mobility = st.slider(
    "🚇 Transportation", 1, 10, 5
)

safety = st.slider(
    "🛡️ Safety", 1, 10, 5
)

social = st.slider(
    "🧑‍🤝‍🧑 Social life", 1, 10, 5
)


# ============================================================
# FIND CITIES
# ============================================================

if st.button("🔍 Find My Cities"):

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        "data/features_processed.csv"
    )


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # DISPLAY NAMES
    # --------------------------------------------------------

    feature_names = {

        "affordability": "Affordability",
        "career": "Career",
        "business": "Business",
        "cafes": "Cafes",
        "fitness": "Fitness",
        "nightlife": "Nightlife",
        "nature": "Nature",
        "climate": "Climate",
        "mobility": "Mobility",
        "safety": "Safety",
        "social": "Social life"
    }


    # --------------------------------------------------------
    # USER PREFERENCES
    # --------------------------------------------------------

    user_preferences = {

        "affordability": affordability,
        "career": career,
        "business": business,
        "cafes": cafes,
        "fitness": fitness,
        "nightlife": nightlife,
        "nature": nature,
        "climate": climate,
        "mobility": mobility,
        "safety": safety,
        "social": social
    }


    # --------------------------------------------------------
    # USER VECTOR
    # --------------------------------------------------------

    user_vector = np.array([
        user_preferences[feature]
        for feature in features
    ]).reshape(1, -1)


    # --------------------------------------------------------
    # CITY MATRIX
    # --------------------------------------------------------

    city_matrix = df[
        features
    ].values


    # --------------------------------------------------------
    # STANDARDIZATION
    # --------------------------------------------------------

    scaler = StandardScaler()

    city_matrix_scaled = scaler.fit_transform(
        city_matrix
    )

    user_vector_scaled = scaler.transform(
        user_vector
    )


    # --------------------------------------------------------
    # CALCULATE SCORES
    # --------------------------------------------------------

    results = []


    for i, (_, city) in enumerate(
        df.iterrows()
    ):

        contributions = {}

        total_score = 0
        total_weight = 0


        # ====================================================
        # COMPATIBILITY SCORE
        # ====================================================

        for feature in features:

            preference = user_preferences[
                feature
            ]

            city_score = city[
                feature
            ]


            # Difference between what user wants
            # and what the city provides

            difference = abs(
                preference - city_score
            )


            # Convert difference into
            # a 1-10 match score

            match_score = (
                10 - difference
            )


            # Important preferences
            # receive more weight

            contribution = (
                preference * match_score
            )


            contributions[
                feature
            ] = contribution


            total_score += contribution

            total_weight += preference


        compatibility = (
            total_score /
            total_weight
        )


        # ====================================================
        # COSINE SIMILARITY
        # ====================================================

        similarity = cosine_similarity(
            user_vector_scaled,
            city_matrix_scaled[
                i
            ].reshape(1, -1)
        )[0][0]


        # Convert -1 to +1
        # into 0 to 10

        similarity_score = (
            (similarity + 1) / 2
        ) * 10


        # ====================================================
        # HYBRID SCORE
        # ====================================================

        hybrid_score = (
            0.8 * compatibility
            +
            0.2 * similarity_score
        )


        # ====================================================
        # STRONGEST / WEAKEST MATCHES
        # ====================================================

        strongest = sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]


        weakest = sorted(
            contributions.items(),
            key=lambda x: x[1]
        )[:3]


        # ====================================================
        # SAVE RESULT
        # ====================================================

        results.append({

            "city": city["city"],

            "compatibility":
                compatibility,

            "similarity":
                similarity_score,

            "hybrid":
                hybrid_score,

            "strongest":
                strongest,

            "weakest":
                weakest
        })


    # --------------------------------------------------------
    # RESULTS DATAFRAME
    # --------------------------------------------------------

    results = pd.DataFrame(
        results
    )


    results = results.sort_values(
        "hybrid",
        ascending=False
    ).reset_index(
        drop=True
    )


    # ========================================================
    # TOP CITY
    # ========================================================

    top_city = results.iloc[0]


    st.divider()

    st.subheader(
        "🏆 Your Top City Match"
    )


    st.markdown(
        f"# 🏙️ {top_city['city']}"
    )


    st.metric(
        "Compatibility Score",
        f"{top_city['hybrid']:.2f} / 10"
    )


    st.progress(
        min(
            top_city["hybrid"] / 10,
            1.0
        )
    )


    st.write(
        f"**Compatibility:** "
        f"{top_city['compatibility']:.2f}/10"
        f"  |  "
        f"**Preference similarity:** "
        f"{top_city['similarity']:.2f}/10"
    )


    # ========================================================
    # STRONGEST MATCHES
    # ========================================================

    st.markdown(
        "### 🟢 Strongest matches"
    )


    strongest_text = " • ".join(
        feature_names[feature]
        for feature, _ in
        top_city["strongest"]
    )


    st.success(
        strongest_text
    )


    # ========================================================
    # POTENTIAL DRAWBACKS
    # ========================================================

    st.markdown(
        "### 🟠 Potential drawbacks"
    )


    weakest_text = " • ".join(
        feature_names[feature]
        for feature, _ in
        top_city["weakest"]
    )


    st.warning(
        weakest_text
    )


    st.divider()


    # ========================================================
    # OTHER CITIES
    # ========================================================

    st.subheader(
        "🏙️ Other Compatible Cities"
    )


    for rank, (_, row) in enumerate(
        results.iloc[1:5].iterrows(),
        start=2
    ):

        st.markdown(
            f"## #{rank} 🏙️ {row['city']}"
        )


        st.metric(
            "Compatibility",
            f"{row['hybrid']:.2f} / 10"
        )


        st.progress(
            min(
                row["hybrid"] / 10,
                1.0
            )
        )


        st.write(
            f"**Compatibility:** "
            f"{row['compatibility']:.2f}/10"
            f"  |  "
            f"**Preference similarity:** "
            f"{row['similarity']:.2f}/10"
        )


        st.divider()


    # ========================================================
    # FEATURE BREAKDOWN
    # ========================================================

    st.subheader(
        "🔎 Feature-by-Feature Breakdown"
    )


    st.write(
        "See how each city performs across "
        "every lifestyle factor."
    )


    selected_city = st.selectbox(
        "Choose a city",
        results["city"].tolist()
    )


    selected_row = df[
        df["city"] == selected_city
    ].iloc[0]


    breakdown = pd.DataFrame({

        "Feature": [
            feature_names[feature]
            for feature in features
        ],

        "City Score": [
            selected_row[feature]
            for feature in features
        ],

        "Your Priority": [
            user_preferences[feature]
            for feature in features
        ]
    })


    st.dataframe(
        breakdown,
        hide_index=True,
        use_container_width=True,

        column_config={

            "City Score":
                st.column_config.ProgressColumn(
                    "🏙️ City Score",
                    min_value=0,
                    max_value=10,
                    format="%.1f"
                ),

            "Your Priority":
                st.column_config.ProgressColumn(
                    "🎯 Your Priority",
                    min_value=0,
                    max_value=10,
                    format="%.0f"
                )
        }
    )


    # ========================================================
    # INDIVIDUAL CITY CHART
    # ========================================================

    st.subheader(
        f"📊 {selected_city} — Feature Scores"
    )


    chart_data = pd.DataFrame({

        "Feature": [
            feature_names[feature]
            for feature in features
        ],

        "Score": [
            selected_row[feature]
            for feature in features
        ]
    })


    fig = px.bar(
        chart_data,

        x="Feature",

        y="Score",

        range_y=[0, 10],

        title=f"{selected_city} Lifestyle Scores"
    )


    fig.update_layout(
        xaxis_title="",
        yaxis_title="Score"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.divider()


    # ========================================================
    # TOP 5 COMPARISON
    # ========================================================

    st.subheader(
        "📈 Compare Top 5 Cities"
    )


    top_cities = results.head(5)[
        "city"
    ].tolist()


    comparison_data = df[
        df["city"].isin(
            top_cities
        )
    ][
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


    comparison_data = comparison_data.melt(
        id_vars="city",
        var_name="feature",
        value_name="score"
    )


    comparison_data["feature"] = (
        comparison_data["feature"]
        .map(feature_names)
    )


    fig2 = px.bar(

        comparison_data,

        x="feature",

        y="score",

        color="city",

        barmode="group",

        range_y=[0, 10],

        title="Top 5 Cities — Lifestyle Comparison"
    )


    fig2.update_layout(
        xaxis_title="",
        yaxis_title="Score",
        legend_title="City"
    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )


    # ========================================================
    # FOOTER
    # ========================================================

    st.divider()


    st.caption(
        "💡 Compatibility is based on your stated "
        "priorities and processed city feature scores."
    )

    st.caption(
        "⚙️ Hybrid score = 80% compatibility + "
        "20% preference similarity."
    )