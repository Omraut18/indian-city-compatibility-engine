import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Indian City Compatibility Engine",
    page_icon="🏙️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #0B0D12;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 4rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #F8FAFC !important;
    }

    p {
        color: #A7AFBD;
    }

    /* Hero */

    .hero-label {
        color: #A78BFA;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        color: #F8FAFC;
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 0.5rem;
    }

    .hero-text {
        color: #A7AFBD;
        font-size: 1rem;
        margin-bottom: 2.5rem;
    }

    /* Section headings */

    .section-title {
        color: #F8FAFC;
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 1rem;
        margin-bottom: 0.15rem;
    }

    .section-subtitle {
        color: #7F8796;
        font-size: 0.85rem;
        margin-bottom: 1.2rem;
    }

    /* Slider */

    div[data-baseweb="slider"] [role="slider"] {
        background-color: #8B5CF6 !important;
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.18) !important;
    }

    /* Button */

    .stButton > button {
        background: linear-gradient(
            135deg,
            #8B5CF6,
            #6366F1
        ) !important;

        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 750 !important;
        padding: 0.65rem 1.25rem !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.25);
    }

    .stButton > button:hover {
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.35);
    }

    /* Result */

    .result-city {
        color: #F8FAFC;
        font-size: 2rem;
        font-weight: 850;
    }

    .result-score {
        color: #A78BFA;
        font-size: 2.4rem;
        font-weight: 850;
    }

    .muted {
        color: #7F8796;
        font-size: 0.82rem;
    }

    hr {
        border-color: #242936 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/features_processed.csv"
    )


df = load_data()


# ============================================================
# FEATURES
# ============================================================

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


feature_labels = {
    "affordability": "💰 Affordability",
    "career": "💼 Career opportunities",
    "business": "🚀 Business / Startups",
    "cafes": "☕ Cafes & lifestyle",
    "fitness": "🏋️ Fitness",
    "nightlife": "🌃 Nightlife",
    "nature": "🌿 Nature",
    "climate": "🌤️ Climate",
    "mobility": "🚇 Transportation",
    "safety": "🛡️ Safety",
    "social": "🧑‍🤝‍🧑 Social life"
}


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero-label">AI-POWERED CITY RECOMMENDER</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">🏙️ Indian City Compatibility Engine</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-text">'
    'Find Indian cities that match the life you actually want.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LIFESTYLE PRIORITIES
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


if "preferences" not in st.session_state:

    st.session_state.preferences = {
        feature: 5.0
        for feature in features
    }


left_features = features[:6]
right_features = features[6:]


col1, col2 = st.columns(
    2,
    gap="large"
)


# ============================================================
# LEFT SLIDERS
# ============================================================

with col1:

    for feature in left_features:

        value = st.slider(
            feature_labels[feature],
            min_value=1.0,
            max_value=10.0,
            value=float(
                st.session_state.preferences[feature]
            ),
            step=1.0,
            key=f"slider_{feature}"
        )

        st.session_state.preferences[feature] = value


# ============================================================
# RIGHT SLIDERS
# ============================================================

with col2:

    for feature in right_features:

        value = st.slider(
            feature_labels[feature],
            min_value=1.0,
            max_value=10.0,
            value=float(
                st.session_state.preferences[feature]
            ),
            step=1.0,
            key=f"slider_{feature}"
        )

        st.session_state.preferences[feature] = value


st.markdown(
    '<div class="muted">'
    '1 = Not important • 10 = Extremely important'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# ============================================================
# RUN ENGINE
# ============================================================

run_engine = st.button(
    "✨ Find My City Matches",
    type="primary"
)


if run_engine:

    preferences = st.session_state.preferences


    # --------------------------------------------------------
    # USER VECTOR
    # --------------------------------------------------------

    user_vector = np.array(
        [
            preferences[feature]
            for feature in features
        ]
    ).reshape(1, -1)


    # --------------------------------------------------------
    # CITY MATRIX
    # --------------------------------------------------------

    city_matrix = df[features].values


    # --------------------------------------------------------
    # STANDARDIZE
    # --------------------------------------------------------

    scaler = StandardScaler()

    city_matrix_scaled = scaler.fit_transform(
        city_matrix
    )

    user_vector_scaled = scaler.transform(
        user_vector
    )


    # --------------------------------------------------------
    # CALCULATE CITY SCORES
    # --------------------------------------------------------

    results = []


    for i, (_, city) in enumerate(
        df.iterrows()
    ):

        total_score = 0
        total_weight = 0

        contributions = {}


        # Weighted compatibility

        for feature in features:

            preference = preferences[feature]
            city_score = city[feature]

            match_score = (
                10 - abs(
                    preference - city_score
                )
            )

            contribution = (
                preference * match_score
            )

            total_score += contribution
            total_weight += preference

            contributions[feature] = contribution


        compatibility = (
            total_score / total_weight
        )


        # Cosine similarity

        similarity = cosine_similarity(
            user_vector_scaled,
            city_matrix_scaled[i].reshape(1, -1)
        )[0][0]


        similarity_score = (
            (similarity + 1) / 2
        ) * 10


        # Hybrid score
        # 80% compatibility
        # 20% similarity

        hybrid_score = (
            0.8 * compatibility
            + 0.2 * similarity_score
        )


        results.append(
            {
                "city": city["city"],
                "compatibility": compatibility,
                "similarity": similarity_score,
                "hybrid": hybrid_score,
                "contributions": contributions
            }
        )


    # --------------------------------------------------------
    # SORT RESULTS
    # --------------------------------------------------------

    results = sorted(
        results,
        key=lambda x: x["hybrid"],
        reverse=True
    )


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🏆 Your city matches</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Based on the lifestyle priorities you selected.'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # TOP CITY
    # ========================================================

    top = results[0]


    top_col1, top_col2 = st.columns(
        [3, 1]
    )


    with top_col1:

        st.markdown(
            '<div class="muted">'
            'TOP COMPATIBILITY MATCH'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-city">'
            f'🏙️ {top["city"]}'
            f'</div>',
            unsafe_allow_html=True
        )


    with top_col2:

        st.markdown(
            '<div class="muted">HYBRID SCORE</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-score">'
            f'{top["hybrid"]:.2f}/10'
            f'</div>',
            unsafe_allow_html=True
        )


    st.write("")


    score1, score2 = st.columns(2)


    with score1:

        st.metric(
            "Weighted compatibility",
            f'{top["compatibility"]:.2f}/10'
        )


    with score2:

        st.metric(
            "Cosine similarity",
            f'{top["similarity"]:.2f}/10'
        )


    # ========================================================
    # STRONGEST / WEAKEST
    # ========================================================

    contributions = top["contributions"]


    strongest = sorted(
        contributions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]


    weakest = sorted(
        contributions.items(),
        key=lambda x: x[1]
    )[:3]


    st.write("")


    strong_col, weak_col = st.columns(2)


    with strong_col:

        st.subheader("🔥 Strongest matches")

        for feature, _ in strongest:

            st.write(
                f"• {feature_labels[feature]}"
            )


    with weak_col:

        st.subheader("⚠️ Potential trade-offs")

        for feature, _ in weakest:

            st.write(
                f"• {feature_labels[feature]}"
            )


    # ========================================================
    # RANKING TABLE
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📊 City rankings</div>',
        unsafe_allow_html=True
    )


    ranking_data = []


    for rank, result in enumerate(
        results,
        start=1
    ):

        ranking_data.append(
            {
                "Rank": rank,
                "City": result["city"],
                "Hybrid score": round(
                    result["hybrid"],
                    2
                ),
                "Compatibility": round(
                    result["compatibility"],
                    2
                ),
                "Similarity": round(
                    result["similarity"],
                    2
                )
            }
        )


    ranking_df = pd.DataFrame(
        ranking_data
    )


    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # TOP 5 CHART
    # ========================================================

    st.write("")


    top5 = ranking_df.head(5)


    fig = px.bar(
        top5,
        x="City",
        y="Hybrid score",
        text="Hybrid score",
        title="Top 5 city compatibility scores"
    )


    fig.update_traces(
        marker_color="#8B5CF6",
        textposition="outside"
    )


    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#F8FAFC"
        ),
        yaxis=dict(
            range=[0, 10],
            gridcolor="#252A35"
        ),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0)"
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # CITY EXPLORER
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🔎 Explore a city</div>',
        unsafe_allow_html=True
    )


    city_names = [
        result["city"]
        for result in results
    ]


    selected_city = st.selectbox(
        "Choose a city",
        city_names
    )


    selected_row = df[
        df["city"] == selected_city
    ].iloc[0]


    breakdown = pd.DataFrame(
        {
            "Feature": [
                feature_labels[feature]
                for feature in features
            ],
            "Score": [
                selected_row[feature]
                for feature in features
            ]
        }
    )


    fig2 = px.bar(
        breakdown,
        x="Score",
        y="Feature",
        orientation="h",
        text="Score",
        title=f"{selected_city} — feature profile"
    )


    fig2.update_traces(
        marker_color="#6366F1",
        texttemplate="%{text:.1f}",
        textposition="outside"
    )


    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#F8FAFC"
        ),
        xaxis=dict(
            range=[0, 10],
            gridcolor="#252A35"
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0)"
        ),
        margin=dict(
            l=20,
            r=50,
            t=60,
            b=20
        )
    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )


    # ========================================================
    # USER PROFILE
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🎯 Your preference profile</div>',
        unsafe_allow_html=True
    )


    preference_df = pd.DataFrame(
        {
            "Feature": [
                feature_labels[feature]
                for feature in features
            ],
            "Importance": [
                preferences[feature]
                for feature in features
            ]
        }
    )


    st.dataframe(
        preference_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.write("")

st.divider()

st.caption(
    "AI-powered lifestyle compatibility engine • "
    "Python • Pandas • Scikit-learn • Plotly • Streamlit"
)