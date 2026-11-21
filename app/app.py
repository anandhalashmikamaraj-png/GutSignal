import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


class PrevalenceFilter(BaseEstimator, TransformerMixin):
    """Required to deserialize the saved GutSignal pipeline."""
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


st.set_page_config(page_title="GutSignal", page_icon="🧬", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "gutsignal_random_forest.pkl"))
    with open(os.path.join(MODEL_DIR, "input_features.json"), encoding="utf-8") as f:
        features = json.load(f)
    return model, [str(x) for x in features]


try:
    model, input_features = load_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()


SAMPLE_ID_NAMES = {
    "sample_id", "sample", "patient_id", "patient", "subject_id",
    "subject", "sampleid", "patientid", "id"
}


def clean_column_names(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if len(df.columns) != len(set(df.columns)):
        raise ValueError("The CSV contains duplicate column names.")
    return df


def prepare_wide(df, features):
    """One row per sample; microbial MSP IDs are columns."""
    df = clean_column_names(df)
    feature_set = set(features)
    matched_cols = [c for c in df.columns if c in feature_set]

    if not matched_cols:
        raise ValueError(
            "I could not find any model MSP IDs as column names. "
            "For wide format, MSP IDs must be column headers."
        )

    id_cols = [
        c for c in df.columns
        if c.strip().lower() in SAMPLE_ID_NAMES and c not in feature_set
    ]
    if len(id_cols) > 1:
        raise ValueError("More than one possible sample-ID column was found.")

    if id_cols:
        sample_ids = df[id_cols[0]].astype(str).str.strip()
        if sample_ids.isna().any() or (sample_ids == "").any():
            raise ValueError("Sample IDs cannot be blank.")
    else:
        sample_ids = pd.Series(
            [f"Sample_{i+1}" for i in range(len(df))], index=df.index
        )

    values = df[matched_cols].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError(
            "Some recognised MSP columns contain blank or non-numeric values. "
            "Clean these values before uploading."
        )
    arr = values.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
