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
        raise ValueError("Abundance values must be finite.")
    if (arr < 0).any():
        raise ValueError("Negative abundance values are not allowed.")

    aligned = pd.DataFrame(0.0, index=df.index, columns=features)
    aligned.loc[:, matched_cols] = values
    missing = [f for f in features if f not in matched_cols]
    return aligned, sample_ids.tolist(), len(matched_cols), missing, len(df)


def prepare_long(df, features):
    """One row per sample–MSP pair; supports single- and multi-sample files."""
    df = clean_column_names(df)
    lower_to_original = {c.lower(): c for c in df.columns}

    if "msp_id" not in lower_to_original or "abundance" not in lower_to_original:
        raise ValueError("Long format requires columns named 'msp_id' and 'abundance'.")

    msp_col = lower_to_original["msp_id"]
    abundance_col = lower_to_original["abundance"]
    sample_candidates = [
        c for c in df.columns
        if c.strip().lower() in SAMPLE_ID_NAMES and c not in {msp_col, abundance_col}
    ]
    if len(sample_candidates) > 1:
        raise ValueError("More than one possible sample-ID column was found.")

    work = df[[msp_col, abundance_col] + sample_candidates].copy()
    work[msp_col] = work[msp_col].astype(str).str.strip()
    if work[msp_col].isna().any() or (work[msp_col] == "").any():
        raise ValueError("MSP IDs cannot be blank.")

    work[abundance_col] = pd.to_numeric(work[abundance_col], errors="coerce")
    if work[abundance_col].isna().any():
        raise ValueError("Abundance values must be numeric and cannot be blank.")
    vals = work[abundance_col].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("Abundance values must be finite.")
    if (vals < 0).any():
        raise ValueError("Negative abundance values are not allowed.")

    if sample_candidates:
        sample_col = sample_candidates[0]
        work[sample_col] = work[sample_col].astype(str).str.strip()
        if work[sample_col].isna().any() or (work[sample_col] == "").any():
            raise ValueError("Sample IDs cannot be blank.")
    else:
        sample_col = "__sample_id__"
        work[sample_col] = "Sample_1"

    recognised = work[work[msp_col].isin(features)].copy()
    if recognised.empty:
        raise ValueError("No uploaded MSP IDs match the trained model's features.")

    duplicate_pairs = recognised.duplicated([sample_col, msp_col], keep=False)
    if duplicate_pairs.any():
        raise ValueError(
            "Duplicate sample–MSP combinations were found. "
            "Each sample should have only one abundance value per MSP."
        )

    pivot = recognised.pivot(index=sample_col, columns=msp_col, values=abundance_col)
    pivot = pivot.reindex(columns=features)
    missing = [f for f in features if f not in recognised[msp_col].unique()]
    pivot = pivot.fillna(0.0)
    sample_ids = [str(x) for x in pivot.index.tolist()]
    pivot.columns.name = None
    return pivot, sample_ids, len(set(recognised[msp_col])), missing, len(pivot)


st.title("🧬 GutSignal")
st.subheader("Gut microbiome research-model explorer")
st.write(
    "Upload microbial abundance data in either wide or long CSV format. "
    "The app maps recognised MSP identifiers to the features used by the trained model."
)
st.warning(
    "Research prototype only. GutSignal does not diagnose colorectal cancer "
    "or determine an individual's clinical risk."
)

with st.expander("CSV formats accepted", expanded=True):
    st.markdown(
        "**Wide format:** one row per sample; MSP identifiers are column names. "
        "An optional `sample_id` or `patient_id` column can identify samples."
    )
    st.code(
        "sample_id,msp_0001,msp_0002,msp_0003\n"
        "P001,0.12,0.00,0.35\n"
        "P002,0.04,0.21,0.00",
        language="csv"
    )
    st.markdown(
        "**Long format:** one row per sample–MSP pair; requires `msp_id` and "
        "`abundance`. Add `sample_id` (or `patient_id`) for multiple samples. "
        "If no sample ID column is supplied, the file is treated as one sample."
    )
    st.code(
        "sample_id,msp_id,abundance\n"
        "P001,msp_0001,0.12\n"
        "P001,msp_0002,0.00\n"
        "P002,msp_0001,0.04",
        language="csv"
    )

st.caption(
    "Age and sex are not used by this trained model. Predictions depend only "
    "on the microbial abundance features."
)

uploaded_file = st.file_uploader("Upload microbiome CSV", type=["csv"])

if uploaded_file is not None:
    try:
        raw = pd.read_csv(uploaded_file)
        if raw.empty:
            raise ValueError("The uploaded CSV is empty.")

        cols_lower = {str(c).strip().lower() for c in raw.columns}
        is_long = {"msp_id", "abundance"}.issubset(cols_lower)

        if is_long:
            X_input, sample_ids, matched_count, missing_features, sample_count = (
                prepare_long(raw, input_features)
            )
            detected_format = "Long"
        else:
            X_input, sample_ids, matched_count, missing_features, sample_count = (
                prepare_wide(raw, input_features)
            )
            detected_format = "Wide"

        st.success(f"{detected_format}-format CSV recognised.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Samples found", sample_count)
        c2.metric("Recognised MSPs", f"{matched_count:,} / {len(input_features):,}")
        c3.metric("Missing model MSPs", f"{len(missing_features):,}")

        if missing_features:
            st.warning(
                f"{len(missing_features)} model MSP features are absent from this file. "
                "They will be set to zero only if the uploaded dataset represents "
                "a complete feature profile and absent MSPs genuinely mean undetected. "
                "Do not continue if the file is truncated or uses a different feature definition."
            )
            confirm_missing = st.checkbox(
                "I confirm that absent model MSPs represent undetected/zero-abundance features."
            )
        else:
            confirm_missing = True

        st.write("Processed model-input preview:")
        st.dataframe(X_input.head(), use_container_width=True)

        if st.button(
            "Analyse uploaded samples",
            type="primary",
            disabled=not confirm_missing
        ):
            predictions = model.predict(X_input[input_features])
            probabilities = model.predict_proba(X_input[input_features])

            results = pd.DataFrame({
                "sample_id": sample_ids,
                "model_classification": [
                    "CRC-like" if p == 1 else "Healthy-like" for p in predictions
                ],
                "healthy_class_score_percent": np.round(probabilities[:, 0] * 100, 2),
                "crc_class_score_percent": np.round(probabilities[:, 1] * 100, 2),
            })

            st.divider()
            st.header("Research-model results")
            st.dataframe(results, use_container_width=True)
            st.download_button(
                "Download prediction results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="GutSignal_predictions.csv",
                mime="text/csv"
            )
            st.info(
                "Scores are model-estimated class probabilities, not an individual's "
                "probability of having cancer. Results are research outputs, not "
                "clinical diagnoses. Input data should use compatible MSP definitions "
                "and abundance processing with the training dataset."
            )

    except Exception as exc:
        st.error(f"Could not process this CSV: {exc}")
