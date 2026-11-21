
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib

from sklearn.base import BaseEstimator, TransformerMixin


# Required for loading the saved model
class PrevalenceFilter(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def fit(self, X, y=None):
        prevalence = (X > 0).mean(axis=0)
        self.selected_features_ = prevalence[
            prevalence >= self.threshold
        ].index.tolist()
        return self

    def transform(self, X):
        return X.loc[:, self.selected_features_]


# Application configuration
st.set_page_config(
    page_title="GutSignal",
    page_icon="🧬",
    layout="wide"
)

# Paths work both locally and on Streamlit Cloud
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")


@st.cache_resource
def load_model():

    model = joblib.load(
        os.path.join(MODEL_DIR, "gutsignal_random_forest.pkl")
    )

    with open(
        os.path.join(MODEL_DIR, "input_features.json")
    ) as f:
        features = json.load(f)

    return model, features


try:
    model, input_features = load_model()

except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()


# Application interface
st.title("🧬 GutSignal")

st.subheader(
    "AI-Powered Gut Microbiome Assessment"
)

st.write(
    "Explore your gut microbiome using a machine-learning "
    "model trained on public colorectal cancer and "
    "healthy-control microbiome cohorts."
)

st.warning(
    "Research prototype only. This model does not diagnose "
    "colorectal cancer or determine an individual's clinical risk."
)

st.divider()


# User information
st.header("1. Your Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

with col2:
    sex = st.selectbox(
        "Sex",
        ["Prefer not to say", "Female", "Male", "Other"]
    )

st.caption(
    "Age and sex are collected for the profile only. "
    "They are not used by the trained prediction model."
)

st.divider()


# Upload microbiome data
st.header("2. Upload Your Gut Microbiome Data")

st.write(
    "Upload a CSV containing two columns: msp_id and abundance. "
    "Use the same MSP identifiers and abundance processing "
    "format as the GutSignal training dataset. Include all "
    "2,425 model input MSPs, using zero for undetected species."
)

uploaded_file = st.file_uploader(
    "Choose your microbiome CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        required_columns = {"msp_id", "abundance"}

        if not required_columns.issubset(data.columns):

            st.error(
                "Your CSV must contain columns named "
                "'msp_id' and 'abundance'."
            )

        else:

            data["msp_id"] = (
                data["msp_id"].astype(str).str.strip()
            )

            data["abundance"] = pd.to_numeric(
                data["abundance"],
                errors="coerce"
            )

            if data["msp_id"].duplicated().any():

                st.error("Duplicate MSP identifiers found.")

            elif data["abundance"].isna().any():

                st.error("Abundance values must be numeric.")

            elif not np.isfinite(data["abundance"]).all():

                st.error("Abundance values must be finite.")

            elif (data["abundance"] < 0).any():

                st.error("Negative abundance values are not allowed.")

            else:

                sample_data = data.set_index(
                    "msp_id"
                )["abundance"]

                matching_features = [
                    feature for feature in input_features
                    if feature in sample_data.index
                ]

                missing_features = [
                    feature for feature in input_features
                    if feature not in sample_data.index
                ]

                unknown_features = (
                    set(sample_data.index) - set(input_features)
                )

                st.write("Uploaded data preview:")
                st.dataframe(data.head())

                st.write(
                    f"Recognised MSPs: "
                    f"{len(matching_features)} / "
                    f"{len(input_features)}"
                )

                if unknown_features:
                    st.warning(
                        f"{len(unknown_features)} unknown MSP "
                        "identifiers will be ignored."
                    )

                if missing_features:

                    st.error(
                        f"{len(missing_features)} required MSPs "
                        "are missing. Please provide all 2,425 "
                        "model input MSPs, including zero values "
                        "for undetected species."
                    )

                else:

                    sample = pd.DataFrame(
                        [[sample_data[feature]
                          for feature in input_features]],
                        columns=input_features
                    )

                    if st.button(
                        "Analyse My Microbiome",
                        type="primary"
                    ):

                        prediction = model.predict(sample)[0]

                        probabilities = model.predict_proba(
                            sample
                        )[0]

                        st.divider()

                        st.header("3. Your Research Results")

                        if prediction == 1:
                            st.error(
                                "CRC-like microbiome profile"
                            )
                        else:
                            st.success(
                                "Healthy-like microbiome profile"
                            )

                        col1, col2 = st.columns(2)

                        col1.metric(
                            "Healthy-class model score",
                            f"{probabilities[0] * 100:.2f}%"
                        )

                        col2.metric(
                            "CRC-class model score",
                            f"{probabilities[1] * 100:.2f}%"
                        )

                        st.info(
                            "These are model-estimated class "
                            "probabilities, not an individual's "
                            "probability of having cancer. "
                            "This research model is not a "
                            "clinical diagnostic tool."
                        )

    except Exception as e:

        st.error(
            f"Error processing uploaded file: {e}"
        )
