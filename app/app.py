import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


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


st.set_page_config(page_title="GutSignal", page_icon="🧬", layout="centered")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "gutsignal_random_forest.pkl"))
    with open(os.path.join(MODEL_DIR, "input_features.json"), "r") as f:
        features = json.load(f)
    return model, features


st.title("🧬 GutSignal")
st.caption("Gut microbiome research-model explorer")
st.warning(
    "Research prototype only. This tool does not diagnose colorectal cancer "
    "or determine an individual's clinical risk."
)

try:
    model, input_features = load_model()
except Exception as exc:
    st.error(f"The trained model could not be loaded: {exc}")
    st.stop()

st.subheader("Your details")
age_col, gender_col = st.columns(2)
with age_col:
    age = st.number_input("Age", min_value=1, max_value=120, value=25, step=1)
with gender_col:
    gender = st.selectbox("Gender", ["Select", "Female", "Male", "Other", "Prefer not to say"])
st.caption("Age and gender are recorded as profile details only; the trained model does not use them.")

st.subheader("Upload microbiome data")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

sample_frame = None
missing_count = 0
recognised_count = 0
parse_error = None

if uploaded_file is not None:
    try:
        raw = pd.read_csv(uploaded_file)
        raw.columns = raw.columns.astype(str).str.strip()
        feature_set = set(input_features)
        lower_to_original = {c.lower(): c for c in raw.columns}

        # Long format: one row per MSP, with optional sample identifier.
        if "msp_id" in lower_to_original and "abundance" in lower_to_original:
            msp_col = lower_to_original["msp_id"]
            abundance_col = lower_to_original["abundance"]
            sid_col = next((lower_to_original[k] for k in
                            ["sample_id", "patient_id", "subject_id"]
                            if k in lower_to_original), None)
            work = raw.copy()
            work[msp_col] = work[msp_col].astype(str).str.strip()
            work[abundance_col] = pd.to_numeric(work[abundance_col], errors="coerce")
            if work[abundance_col].isna().any() or not np.isfinite(work[abundance_col]).all():
                raise ValueError("Abundance values must all be numeric and finite.")
            if (work[abundance_col] < 0).any():
                raise ValueError("Negative abundance values are not allowed.")
            if sid_col is None:
                work["__sample_id"] = "Sample_1"
                sid_col = "__sample_id"
            work[sid_col] = work[sid_col].fillna("Sample_1").astype(str)
            if work.duplicated([sid_col, msp_col]).any():
                raise ValueError("Duplicate sample–MSP rows were found. Each sample/MSP pair must appear once.")
            recognised = work[work[msp_col].isin(feature_set)]
            recognised_count = recognised[msp_col].nunique()
            pivot = recognised.pivot(index=sid_col, columns=msp_col, values=abundance_col)
            sample_frame = pivot.reindex(columns=input_features, fill_value=0.0).fillna(0.0)
            missing_count = len(feature_set - set(recognised[msp_col]))
        else:
            # Wide format: one sample per row, MSP IDs in column names.
            id_candidates = [lower_to_original[k] for k in
                             ["sample_id", "patient_id", "subject_id"] if k in lower_to_original]
            id_col = id_candidates[0] if id_candidates else None
            feature_cols = [c for c in raw.columns if c in feature_set]
            if not feature_cols:
                raise ValueError("No recognised MSP columns were found. Use MSP IDs as column names (wide format), or provide msp_id and abundance columns (long format).")
            values = raw[feature_cols].apply(pd.to_numeric, errors="coerce")
            if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
                raise ValueError("All MSP abundance values must be numeric and finite. Remove blank or non-numeric cells.")
            if (values < 0).any().any():
                raise ValueError("Negative abundance values are not allowed.")
            sample_frame = values.copy()
            sample_frame.index = (raw[id_col].fillna("Sample").astype(str).tolist()
                                  if id_col else [f"Sample_{i+1}" for i in range(len(raw))])
            sample_frame = sample_frame.reindex(columns=input_features, fill_value=0.0)
            recognised_count = len(feature_cols)
            missing_count = len(input_features) - recognised_count

        st.success(f"CSV loaded: {len(sample_frame)} sample(s) detected.")
        st.caption(f"Recognised model features: {recognised_count} / {len(input_features)}")
        if missing_count:
            st.warning(
                f"{missing_count} model MSP feature(s) are absent from this file. "
                "They will be treated as zero only if the file is a complete profile "
                "and absent MSPs genuinely mean undetected/zero abundance."
            )
            confirm_missing = st.checkbox("I confirm absent MSPs in this complete profile represent zero abundance.")
        else:
            confirm_missing = True
    except Exception as exc:
        parse_error = str(exc)
        st.error(f"Could not read this CSV: {parse_error}")
        confirm_missing = False
else:
    confirm_missing = False

run_clicked = st.button("Get Result", type="primary", use_container_width=True,
                         disabled=(sample_frame is None or not confirm_missing))

if run_clicked and sample_frame is not None:
    try:
        predictions = model.predict(sample_frame)
        probabilities = model.predict_proba(sample_frame)
        result_rows = []
        for i, (sample_id, prediction, probs) in enumerate(zip(sample_frame.index, predictions, probabilities)):
            result_rows.append({
                "Sample ID": sample_id,
                "Prediction": "CRC-like" if int(prediction) == 1 else "Healthy-like",
                "Healthy model score (%)": round(float(probs[0]) * 100, 2),
                "CRC model score (%)": round(float(probs[1]) * 100, 2),
            })
        results = pd.DataFrame(result_rows)
        st.subheader("Result")
        st.caption(f"Profile details — Age: {age}; Gender: {gender}")
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button("Download results CSV", results.to_csv(index=False),
                           file_name="GutSignal_results.csv", mime="text/csv")
        st.info("Model scores are not an individual's probability of cancer. This research prototype is not a clinical diagnostic tool.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
