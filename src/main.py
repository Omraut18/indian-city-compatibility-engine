import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Indian City Compatibility Engine",
    page_icon="🏙️",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at 50% -10%,
            rgba(99, 102, 241, 0.18),
            transparent 35%
        ),
        #0b0d12;
    color: #f8fafc;
}

.main .block-container {
    max-width: 1050px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}


/* =========================================================
   TEXT
   ========================================================= */

h1, h2, h3, h4 {
    color: #f8fafc !important;
    letter-spacing: -0.02em;
}

p {
    color: #aeb6c5;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    padding: 2.2rem 2.2rem 2rem 2.2rem;
    border-radius: 24px;
    background:
        linear-gradient(
            135deg,
            rgba(99, 102, 241, 0.20),
            rgba(59, 130, 246, 0.08)
        );
    border: 1px solid rgba(129, 140, 248, 0.20);
    margin-bottom: 2rem;
}

.hero-badge {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.16);
    color: #a5b4fc;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
}

.hero-title {
    font-size: 2.7rem;
    font-weight: 800;
    line-height: 1.05;
    color: #ffffff;
    margin-bottom: 0.7rem;
}

.hero-subtitle {
    font-size: 1rem;
    color: #aeb6c5;
}


/* =========================================================
   SECTION HEADERS
   ========================================================= */

.section-title {
    font-size: 1.45rem;
    font-weight: 750;
    color: #f8fafc;
    margin-top: 1rem;
    margin-bottom: 0.3rem;
}

.section-subtitle {
    color: #8993a5;
    font-size: 0.9rem;
    margin-bottom: 1.2rem;
}


/* =========================================================
   PREFERENCE CARD
   ========================================================= */

.preference-card {
    padding: 1.3rem 1.4rem;
    border-radius: 16px;
    background: rgba(22, 27, 34, 0.72);
    border: 1px solid #252b36;
    margin-bottom: 0.8rem;
}


/* =========================================================
   SLIDERS
   ========================================================= */

/* slider label */
[data-testid="stWidgetLabel"] p {
    color: #e5e7eb !important;
    font-weight: 600 !important;
}

/* slider value */
[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
    color: #a5b4fc !important;
}

/* slider thumb */
[data-baseweb="slider"] [role="slider"] {
    background: #8b5cf6 !important;
    border: 3px solid #c4b5fd !important;
    box-shadow:
        0 0 0 4px rgba(139, 92, 246, 0.15),
        0 0 16px rgba(139, 92, 246, 0.35);
}

/* slider track */
[data-baseweb="slider"] > div > div {
    background-color: #292f3a !important;
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {
    width: 100%;
    height: 3.2rem;
    border: 0;
    border-radius: 14px;

    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #4f46e5
        );

    color: white;
    font-size: 1rem;
    font-weight: 750;

    box-shadow:
        0 8px 30px rgba(79, 70, 229, 0.25);

    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    background:
        linear-gradient(
            135deg,
            #8b5cf6,
            #6366f1
        );

    box-shadow:
        0 12px 35px rgba(99, 102, 241, 0.35);
}


/* =========================================================
   TOP CITY CARD
   ========================================================= */

.top-city {
    padding: 2rem;
    border-radius: 22px;

    background:
        linear-gradient(
            135deg,
            rgba(99, 102, 241, 0.18),
            rgba(59, 130, 246, 0.07)
        );

    border: 1px solid rgba(129, 140, 248, 0.25);

    margin-top: 1rem;
    margin-bottom: 1.4rem;

    box-shadow:
        0 15px 50px rgba(0, 0, 0, 0.20);
}

.top-city-label {
    color: #a5b4fc;
    font-size: 0.82rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.top-city-name {
    font-size: 2.7rem;
    font-weight: 850;
    color: white;
    margin: 0.35rem 0;
}

.top-city-score {
    font-size: 1.15rem;
    color: #c4b5fd;
}


/* =========================================================
   SCORE CARDS
   ========================================================= */

.score-card {
    padding: 1.1rem;
    border-radius: 15px;
    background: #12161d;
    border: 1px solid #252b36;
}

.score-label {
    color: #8d97a8;
    font-size: 0.78rem;
}

.score-value {
    color: #f8fafc;
    font-size: 1.35rem;
    font-weight: 750;
}


/* =========================================================
   CITY RANK CARDS
   ========================================================= */

.city-card {
    padding: 1.1rem 1.3rem;
    border-radius: 15px;
    background: #12161d;
    border: 1px solid #252b36;
    margin-bottom: 0.7rem;
}

.city-rank {
    color: #8b5cf6;
    font-weight: 800;
    font-size: 0.8rem;
}

.city-name {
    color: #f8fafc;
    font-size: 1.15rem;
    font-weight: 700;
}

.city-score {
    color: #a5b4fc;
    font-weight: 750;
    font-size: 1.1rem;
}


/* =========================================================
   FEATURE BOXES
   ========================================================= */

.feature-good {
    padding: 1rem 1.2rem;
    border-radius: 14px;
    background: rgba(16, 185, 129, 0.09);
    border: 1px solid rgba(16, 185, 129, 0.18);
    color: #6ee7b7;
    font-weight: 600;
}

.feature-bad {
    padding: 1rem 1.2rem;
    border-radius: 14px;
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.18);
    color: #fcd34d;
    font-weight: 600;
}


/* =========================================================
   SELECT BOX
   ========================================================= */

[data-baseweb="select"] > div {
    background-color: #12161d !important;
    border: 1px solid #303746 !important;
    border-radius: 10px !important;
}


/* =========================================================
   PROGRESS
   ========================================================= */

[data-testid="stProgressBar"] > div > div > div {
    background:
        linear-gradient(
            90deg,
            #7c3aed,
            #4f46e5
        ) !important;
}


/* =========================================================
   DIVIDER
   ========================================================= */

hr {
    border-color: #252b36 !important;
    margin: 2rem 0;
}


/* =========================================================
   DATAFRAME
   ========================================================= */

[data-testid="stDataFrame"] {
    border: 1px solid #252b36;
    border-radius: 14px;
    overflow: hidden;
}


/* =========================================================
   CAPTION
   ========================================================= */

.stCaption {
    color: #667085 !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-badge">
        AI-POWERED CITY RECOMMENDER
    </div>

    <div class="hero-title">
        🏙️ Indian City<br>
        Compatibility Engine
    </div>

    <div class="hero-subtitle">
        Find Indian cities that match the life you actually want.
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# PREFERENCES
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Your lifestyle priorities</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Rate how important each factor is to you.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SLIDERS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    affordability = st.slider(
        "💰 Affordability",
        1, 10, 5
    )

    career = st.slider(
        "💼 Career opportunities",
        1, 10, 5
    )

    business = st.slider(
        "🚀 Business / Startups",
        1, 10, 5
    )

    cafes = st.slider(
        "☕ Cafes & lifestyle",
        1, 10, 5
    )

    fitness = st.slider(
        "🏋️ Fitness",
        1, 10, 5
    )

    nightlife = st.slider(
        "🌃 Nightlife",
        1, 10, 5
    )


with col2:

    nature = st.slider(
        "🌿 Nature",
        1, 10, 5
    )

    climate = st.slider(
        "🌤️ Climate",
        1, 10, 5
    )

    mobility = st.slider(
        "🚇 Transportation",
        1, 10, 5
    )

    safety = st.slider(
        "🛡️ Safety",
        1, 10, 5
    )

    social = st.slider(
        "🧑‍🤝‍🧑 Social life",
        1, 10, 5
    )


st.write("")

st.caption(
    "1 = Not important     •     10 = Extremely important"
)


# ============================================================
# BUTTON
# ============================================================

if st.button("✨ Find My City Matches"):

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = pd.read_csv(
        "data/features_processed.csv"
    )

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


    # ========================================================
    # USER VECTOR
    # ========================================================

    user_vector = np.array([
        user_preferences[feature]
        for feature in features
    ]).reshape(1, -1)

    city_matrix = df[
        features
    ].values

    scaler = StandardScaler()

    city_matrix_scaled = scaler.fit_transform(
        city_matrix
    )

    user_vector_scaled = scaler.transform(
        user_vector
    )


    # ========================================================
    # CALCULATE RESULTS
    # ========================================================

    results = []

    for i, (_, city) in enumerate(
        df.iterrows()
    ):

        contributions = {}

        total_score = 0
        total_weight = 0

        for feature in features:

            preference = user_preferences[
                feature
            ]

            city_score = city[
                feature
            ]

            difference = abs(
                preference - city_score
            )

            match_score = (
                10 - difference
            )

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
        # STRONGEST / WEAKEST
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


        results.append({
            "city": city["city"],
            "compatibility": compatibility,
            "similarity": similarity_score,
            "hybrid": hybrid_score,
            "strongest": strongest,
            "weakest": weakest
        })


    # ========================================================
    # SORT
    # ========================================================

    results = pd.DataFrame(results)

    results = results.sort_values(
        "hybrid",
        ascending=False
    ).reset_index(drop=True)


    top_city = results.iloc[0]


    # ========================================================
    # TOP CITY
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🏆 Your city match</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="top-city">

            <div class="top-city-label">
                TOP COMPATIBILITY
            </div>

            <div class="top-city-name">
                🏙️ {top_city['city']}
            </div>

            <div class="top-city-score">
                {top_city['hybrid']:.2f} / 10 compatibility
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # SCORE CARDS
    # ========================================================

    score1, score2 = st.columns(2)

    with score1:

        st.markdown(
            f"""
            <div class="score-card">

                <div class="score-label">
                    Compatibility
                </div>

                <div class="score-value">
                    {top_city['compatibility']:.2f}/10
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with score2:

        st.markdown(
            f"""
            <div class="score-card">

                <div class="score-label">
                    Preference similarity
                </div>

                <div class="score-value">
                    {top_city['similarity']:.2f}/10
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    # ========================================================
    # STRONGEST / WEAKEST
    # ========================================================

    strongest_text = " • ".join(
        feature_names[feature]
        for feature, _
        in top_city["strongest"]
    )

    weakest_text = " • ".join(
        feature_names[feature]
        for feature, _
        in top_city["weakest"]
    )


    good1, good2 = st.columns(2)

    with good1:

        st.markdown(
            f"""
            <div class="feature-good">

                🟢 Strongest matches<br><br>

                {strongest_text}

            </div>
            """,
            unsafe_allow_html=True
        )

    with good2:

        st.markdown(
            f"""
            <div class="feature-bad">

                🟠 Potential drawbacks<br><br>

                {weakest_text}

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # OTHER CITIES
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🏙️ Other compatible cities</div>',
        unsafe_allow_html=True
    )

    for rank, (_, row) in enumerate(
        results.iloc[1:6].iterrows(),
        start=2
    ):

        st.markdown(
            f"""
            <div class="city-card">

                <span class="city-rank">
                    #{rank}
                </span>

                &nbsp;&nbsp;

                <span class="city-name">
                    {row['city']}
                </span>

                <span style="float:right"
                      class="city-score">
                    {row['hybrid']:.2f}/10
                </span>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # FEATURE BREAKDOWN
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🔎 Feature breakdown</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Compare a city against your personal priorities.'
        '</div>',
        unsafe_allow_html=True
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
    # CITY CHART
    # ========================================================

    st.markdown(
        f'<div class="section-title">'
        f'📊 {selected_city} lifestyle profile'
        f'</div>',
        unsafe_allow_html=True
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
        range_y=[0, 10]
    )


    fig.update_traces(
        marker_color="#8B5CF6"
    )


    fig.update_layout(

        paper_bgcolor="#0b0d12",
        plot_bgcolor="#0b0d12",

        font=dict(
            color="#dbe3ef"
        ),

        xaxis=dict(
            title="",
            gridcolor="#252b36"
        ),

        yaxis=dict(
            title="Score",
            gridcolor="#252b36"
        ),

        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # TOP 5 COMPARISON
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📈 Compare top cities</div>',
        unsafe_allow_html=True
    )

    top_cities = results.head(5)[
        "city"
    ].tolist()


    comparison_data = df[
        df["city"].isin(top_cities)
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

        color_discrete_sequence=[
            "#8B5CF6",
            "#6366F1",
            "#3B82F6",
            "#06B6D4",
            "#22C55E"
        ]
    )


    fig2.update_layout(

        paper_bgcolor="#0b0d12",
        plot_bgcolor="#0b0d12",

        font=dict(
            color="#dbe3ef"
        ),

        xaxis=dict(
            title="",
            gridcolor="#252b36"
        ),

        yaxis=dict(
            title="Score",
            gridcolor="#252b36"
        ),

        legend=dict(
            title="City"
        ),

        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        )
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
        "⚙️ Hybrid score = 80% compatibility + "
        "20% preference similarity."
    )

    st.caption(
        "Data-driven portfolio prototype • "
        "City scores may include estimated lifestyle metrics."
    )