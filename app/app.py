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

