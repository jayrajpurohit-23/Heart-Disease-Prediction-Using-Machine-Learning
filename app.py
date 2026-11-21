
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="",
    layout="wide"
)

# Load saved model
MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "heart_disease_pipeline.pkl"
)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

model = load_model()

# Application heading
st.title("Heart Disease Prediction System")

st.markdown("""
### Machine Learning Mini Project

This application uses a trained machine-learning
pipeline to predict a binary heart disease class
from the supplied dataset attributes.
""")

st.warning(
    "Educational prototype only. "
    "This application is not a medical diagnostic tool."
)

st.divider()

st.subheader("Enter Patient Attributes")

# Input form
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=45
        )

        sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x:
                "Female (0)" if x == 0 else "Male (1)"
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [1, 2, 3, 4]
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=80,
            max_value=220,
            value=120
        )

        chol = st.number_input(
            "Serum Cholesterol",
            min_value=100,
            max_value=600,
            value=200
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [0, 1]
        )

        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2]
        )

    with col2:

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=60,
            max_value=220,
            value=150
        )

        exang = st.selectbox(
            "Exercise Induced Angina",
            [0, 1]
        )

        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=7.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "Slope",
            [1, 2, 3]
        )

        ca = st.selectbox(
            "Major Vessels (CA)",
            [0, 1, 2, 3]
        )

        thal = st.selectbox(
            "Thal",
            [3, 6, 7]
        )

    submit = st.form_submit_button(
        "Predict Heart Disease",
        use_container_width=True
    )

# Prediction
if submit:

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "Model Prediction: Heart Disease Present"
        )

    else:

        st.success(
            "Model Prediction: No Heart Disease Indicated"
        )

    st.write(
        "Model-estimated class probabilities "
        "(not calibrated clinical risk estimates):"
    )

    result = pd.DataFrame({
        "Class": [
            "No Heart Disease",
            "Heart Disease Present"
        ],
        "Model Probability (%)": [
            round(probability[0] * 100, 2),
            round(probability[1] * 100, 2)
        ]
    })

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "This is an educational machine-learning "
        "prediction, not a diagnosis. Consult a "
        "qualified healthcare professional for "
        "medical assessment."
    )

st.divider()

st.caption(
    "Jay Rajpurohit | "
    "Gagan Chouhan | Himanshu Jangid"
)
