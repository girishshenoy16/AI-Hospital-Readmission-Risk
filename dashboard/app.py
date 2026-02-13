import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import shap


RISK_COLORS = {
    "Very High Risk": "#c0392b",  # dark red
    "High Risk": "#e67e22",       # orange
    "Medium Risk": "#f1c40f",     # amber
    "Low Risk": "#27ae60"         # green
}


def style_table(df, highlight_col="Risk Category"):
    def highlight_row(row):
        color = RISK_COLORS.get(row[highlight_col], "")
        if color:
            return [f"background-color: {color}20"] * len(row)
        return [""] * len(row)

    return (
        df.style
        .apply(highlight_row, axis=1)
        .set_properties(**{
            "text-align": "center",
            "font-size": "14px"
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("text-align", "center"),
                ("font-weight", "600")
            ]},
            {"selector": "td", "props": [
                ("text-align", "center"),
                ("padding", "8px")
            ]}
        ])
    )



# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Hospital Readmission Risk Dashboard",
    layout="wide"
)

st.title("🏥 Hospital Readmission Risk Analytics")
st.markdown(
    """
    This dashboard helps care teams **identify high-risk patients**
    for 30-day readmission using an AI-based risk model.
    """
)

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

@st.cache_data
def load_data(data_path):
    return pd.read_csv(data_path)


model = load_model("./models/xgb_classifier_readmission_model.pkl")
data = load_data("./data/processed/engineered_dataset.csv")

@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_shap_explainer(model)

X = data.drop(columns=["readmitted_flag"])
y = data["readmitted_flag"]


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Risk Configuration")

threshold = st.sidebar.slider(
    "Readmission Risk Threshold",
    min_value=0.05,
    max_value=0.80,
    value=0.25,
    step=0.05
)

st.sidebar.markdown(
    """
    **Lower threshold → higher recall**  
    **Higher threshold → fewer false alarms**
    """
)

# -----------------------------
# Model predictions
# -----------------------------
risk_scores = model.predict_proba(X)[:, 1]

results = X.copy()
results["readmission_risk"] = risk_scores
results["risk_flag"] = (results["readmission_risk"] >= threshold).astype(int)

# -----------------------------
# KPIs
# -----------------------------
total_patients = len(results)
high_risk_patients = results["risk_flag"].sum()
avg_risk = results["readmission_risk"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("Total Patients", total_patients)
col2.metric("High-Risk Patients", int(high_risk_patients))
col3.metric("Average Risk Score", f"{avg_risk:.2f}")

# -----------------------------
# Risk distribution
# -----------------------------
st.subheader("📊 Readmission Risk Distribution")

def assign_risk_category(score):
    if score >= 0.85:
        return "Very High Risk"
    elif score >= 0.65:
        return "High Risk"
    elif score >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"

results["Risk Category"] = results["readmission_risk"].apply(assign_risk_category)

risk_order = [
    "Very High Risk",
    "High Risk",
    "Medium Risk",
    "Low Risk"
]

risk_counts = (
    results["Risk Category"]
    .value_counts()
    .reindex(risk_order)
    .fillna(0)
)

# --- Matplotlib plot ---
fig, ax = plt.subplots(figsize=(10, 4))

bar_colors = [RISK_COLORS[risk] for risk in risk_counts.index]

ax.barh(
    risk_counts.index,
    risk_counts.values,
    color=bar_colors
)


ax.set_xlabel("Patient Count")
ax.set_ylabel("Risk Category")
ax.set_title(f"Risk Distribution (Threshold = {threshold:.2f})")

ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.invert_yaxis()  # High risk on top

plt.tight_layout()
st.pyplot(fig)


# -----------------------------------
# Optional: Show underlying data table
# -----------------------------------
with st.expander("📋 View underlying risk data"):
    st.markdown("### 🧮 Patient Count by Risk Category")


    results["Risk Category"] = pd.qcut(
        results["readmission_risk"],
        q=[0, 0.60, 0.80, 0.95, 1.0],
        labels=[
            "Low Risk",
            "Medium Risk",
            "High Risk",
            "Very High Risk"
        ]
    )

    st.write(
        results["Risk Category"].value_counts()
    )

    st.subheader("🔍 Patient-Level Risk Scores")
    st.markdown(
        "Patients ranked by **predicted 30-day readmission risk**. "
        "This view is intended for prioritization and review."
    )

    # Number of patients to show
    top_n = st.slider(
        "Number of patients to display",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )

    # Prepare data
    patient_df = (
        results
        .sort_values("readmission_risk", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    # Add presentation columns
    patient_df["Priority Rank"] = patient_df.index + 1

    patient_df["Readmission Risk (%)"] = (
            patient_df["readmission_risk"] * 100
    ).round(1)

    # Final display dataframe
    display_patient_df = patient_df[
        ["Priority Rank", "Readmission Risk (%)", "Risk Category"]
    ].reset_index(drop=True)

    # Display (no index, clean values)
    st.dataframe(
        style_table(display_patient_df, highlight_col="Risk Category")
        .format({"Readmission Risk (%)": "{:.1f}%"}),
        use_container_width=True,
        height=360
    )

# -----------------------------
# High-risk patient table
# -----------------------------
st.subheader("🚨 High-Risk Patient List")
st.markdown("Patients ranked by **highest predicted readmission risk**.")
st.markdown(
    "Top patients ranked by **predicted 30-day readmission risk**. "
    "This view helps care teams prioritize follow-up actions."
    "(Patients ranked by **highest predicted readmission risk**)"
)

# Select number of patients to display
top_n = st.slider(
    "Number of high-risk patients to display",
    min_value=5,
    max_value=200,
    value=20,
    step=10
)

# Prepare data
high_risk_df = (
    results
    .sort_values("readmission_risk", ascending=False)
    .head(top_n)
    .reset_index(drop=True)
)

high_risk_df["Rank"] = high_risk_df.index + 1
high_risk_df["Readmission Risk (%)"] = (high_risk_df["readmission_risk"] * 100).round(1)

show_all = st.checkbox("Show patients across all risk levels", value=False)

# Select rows FIRST
if show_all:
    high_risk_df = (
        results
        .sample(top_n, random_state=42)
        .reset_index(drop=True)
    )
else:
    high_risk_df = (
        results
        .sort_values("readmission_risk", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

# THEN add derived columns
high_risk_df["Rank"] = high_risk_df.index + 1
high_risk_df["Readmission Risk (%)"] = (
    high_risk_df["readmission_risk"] * 100
).round(1)

# Final display
display_df = high_risk_df[
    ["Rank", "Readmission Risk (%)", "Risk Category"]
]



# ---------- Styling ----------
def highlight_rows(row):
    if row["Risk Category"] == "High Risk":
        return ["background-color: rgba(255, 99, 71, 0.15);"] * len(row)
    elif row["Risk Category"] == "Medium Risk":
        return ["background-color: rgba(255, 165, 0, 0.12);"] * len(row)
    return [""] * len(row)

table_row_df = display_df.copy()
st.dataframe(
    style_table(table_row_df, highlight_col="Risk Category")
        .format({"Readmission Risk (%)": "{:.1f}%"}),
    use_container_width=True,
    height=420
)


st.subheader("🧠 Why is this patient high-risk?")

patient_idx = st.selectbox(
    "Select a patient rank to explain",
    options=display_df["Rank"].tolist()
)

# Map rank → original row
selected_row = high_risk_df.iloc[patient_idx - 1]
selected_features = X.iloc[[selected_row.name]]

# Compute SHAP values
shap_values = explainer.shap_values(selected_features)

# Plot SHAP explanation
fig, ax = plt.subplots(figsize=(8, 4))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=selected_features.iloc[0],
        feature_names=selected_features.columns
    ),
    show=False
)

plt.tight_layout()
st.pyplot(fig)


# -----------------------------------
# Extract top SHAP drivers for patient
# -----------------------------------
shap_contrib = pd.DataFrame({
    "feature": selected_features.columns,
    "shap_value": shap_values[0]
})

shap_contrib["abs_value"] = shap_contrib["shap_value"].abs()
top_features = (
    shap_contrib
    .sort_values("abs_value", ascending=False)
    .head(3)
)


risk_pct = selected_row["readmission_risk"] * 100
risk_category = selected_row["Risk Category"]

top_feature_names = ", ".join(top_features["feature"].tolist())

if risk_category == "Very High Risk":
    message = (
        f"🔴 **Very High Readmission Risk Detected**\n\n"
        f"This patient has a **{risk_pct:.1f}% predicted risk** of readmission.\n\n"
        f"The model identified **{top_feature_names}** as the most critical contributors "
        f"to readmission risk. Immediate intervention, intensive care coordination, "
        f"and proactive post-discharge follow-up are strongly recommended."
    )

elif risk_category == "High Risk":
    message = (
        f"🚨 **High Readmission Risk Detected**\n\n"
        f"This patient has a **{risk_pct:.1f}% predicted risk** of readmission.\n\n"
        f"The model identified **{top_feature_names}** as the strongest contributors "
        f"increasing readmission risk. Priority follow-up and care coordination "
        f"are recommended."
    )

elif risk_category == "Medium Risk":
    message = (
        f"⚠️ **Moderate Readmission Risk**\n\n"
        f"This patient has a **{risk_pct:.1f}% predicted risk** of readmission.\n\n"
        f"Key influencing factors include **{top_feature_names}**. "
        f"Monitoring and selective intervention may help reduce risk."
    )

else:  # Low Risk
    message = (
        f"✅ **Low Readmission Risk**\n\n"
        f"This patient has a **{risk_pct:.1f}% predicted risk** of readmission.\n\n"
        f"No strong high-risk indicators were detected. Routine follow-up "
        f"is likely sufficient."
    )


# Display personalized Info Block
st.info(message)

# -----------------------------
# Model interpretation note
# -----------------------------
st.info(
    """
    ℹ️ **Interpretation Note**

    This model is designed for **risk stratification**, not binary diagnosis.
    High recall is intentionally prioritized to minimize missed readmissions.
    """
)