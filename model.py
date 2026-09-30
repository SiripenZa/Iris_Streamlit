import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)


# =========================================================
# LOAD DATASET
# =========================================================

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "iris_model.pkl"

try:
    model = joblib.load(MODEL_PATH)

except FileNotFoundError:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    joblib.dump(
        model,
        MODEL_PATH
    )


# =========================================================
# SESSION STATE
# =========================================================

if "predicted" not in st.session_state:
    st.session_state.predicted = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None


# =========================================================
# TITLE
# =========================================================

st.markdown(
    "# 🌸 Iris Flower Classifier"
)

st.markdown(
    "Predict the species of Iris flowers using Machine Learning"
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌸 Iris Learn")

st.sidebar.subheader(
    "📊 Input Features"
)

st.sidebar.caption(
    "Adjust the sliders to input flower measurements."
)


# =========================================================
# SLIDERS
# =========================================================

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    min_value=float(X.iloc[:, 0].min()),
    max_value=float(X.iloc[:, 0].max()),
    value=5.4,
    step=0.1
)


sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    min_value=float(X.iloc[:, 1].min()),
    max_value=float(X.iloc[:, 1].max()),
    value=3.0,
    step=0.1
)


petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    min_value=float(X.iloc[:, 2].min()),
    max_value=float(X.iloc[:, 2].max()),
    value=4.0,
    step=0.1
)


petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    min_value=float(X.iloc[:, 3].min()),
    max_value=float(X.iloc[:, 3].max()),
    value=1.2,
    step=0.1
)


# =========================================================
# PREDICT BUTTON
# =========================================================

predict_button = st.sidebar.button(
    "🎯 Predict Species",
    use_container_width=True
)


# =========================================================
# WHEN BUTTON IS CLICKED
# =========================================================

if predict_button:

    input_data = np.array([
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]).reshape(1, -1)


    # Prediction
    prediction = model.predict(
        input_data
    )[0]


    # Probability
    probabilities = model.predict_proba(
        input_data
    )[0]


    # Save result
    st.session_state.predicted = True

    st.session_state.prediction = prediction

    st.session_state.probabilities = probabilities


# =========================================================
# SHOW RESULT ONLY AFTER CLICK
# =========================================================

if st.session_state.predicted:

    prediction = st.session_state.prediction

    probabilities = st.session_state.probabilities


    # =====================================================
    # SPECIES
    # =====================================================

    species_names = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]


    predicted_species = species_names[
        prediction
    ]


    confidence = (
        probabilities[prediction] * 100
    )


    # =====================================================
    # TWO COLUMNS
    # =====================================================

    left_col, right_col = st.columns(
        [1.15, 0.85]
    )


    # =====================================================
    # LEFT
    # =====================================================

    with left_col:

        st.subheader(
            "📈 Input Visualization"
        )

        st.caption(
            "Your Input vs Dataset Average"
        )


        feature_names = [
            "Sepal Length",
            "Sepal Width",
            "Petal Length",
            "Petal Width"
        ]


        user_values = np.array([
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ])


        dataset_average = X.mean().values


        # Create chart

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )


        x = np.arange(
            len(feature_names)
        )


        width = 0.35


        ax.bar(
            x - width / 2,
            user_values,
            width,
            label="Your Input"
        )


        ax.bar(
            x + width / 2,
            dataset_average,
            width,
            label="Dataset Average"
        )


        ax.set_xticks(x)

        ax.set_xticklabels(
            feature_names,
            fontsize=8
        )


        ax.set_ylabel(
            "Value"
        )


        ax.legend()


        ax.grid(
            axis="y",
            alpha=0.2
        )


        plt.tight_layout()


        st.pyplot(
            fig,
            use_container_width=True
        )


    # =====================================================
    # RIGHT
    # =====================================================

    with right_col:

        st.subheader(
            "🎯 Prediction Result"
        )


        # -------------------------------------------------
        # RESULT BOX
        # -------------------------------------------------

        st.info(
            "Predicted Species"
        )


        st.markdown(
            f"# {predicted_species}"
        )


        st.success(
            f"Confidence: {confidence:.1f}%"
        )


        # -------------------------------------------------
        # PROBABILITY
        # -------------------------------------------------

        st.subheader(
            "Probability Distribution"
        )


        probability_df = pd.DataFrame(
            {
                "Species": [
                    "Setosa",
                    "Versicolor",
                    "Virginica"
                ],

                "Probability (%)":
                    probabilities * 100
            }
        )


        st.bar_chart(
            probability_df.set_index(
                "Species"
            ),
            y="Probability (%)"
        )


    # =====================================================
    # INPUT VALUES
    # =====================================================

    st.divider()

    st.subheader(
        "📋 Input Values"
    )


    input_table = pd.DataFrame(
        {
            "Feature": [
                "Sepal Length",
                "Sepal Width",
                "Petal Length",
                "Petal Width"
            ],

            "Value (cm)": [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]
        }
    )


    st.dataframe(
        input_table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# BEFORE PREDICT
# =========================================================

else:

    st.info(
        "👈 Select the flower measurements "
        "from the left side, then click "
        "**🎯 Predict Species**"
    )