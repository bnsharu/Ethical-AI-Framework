import os
import warnings

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATH
# ============================================================

# app.py is inside:
# Ethical_AI_Framework/dashboard/app.py

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Ethical AI Framework",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 19px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .risk-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
    }

    .risk-high {
        background-color: #4c1d1d;
        border: 1px solid #ef4444;
    }

    .risk-medium {
        background-color: #4a3515;
        border: 1px solid #f59e0b;
    }

    .risk-low {
        background-color: #16351f;
        border: 1px solid #22c55e;
    }

    .risk-title {
        font-size: 18px;
        font-weight: 600;
    }

    .risk-value {
        font-size: 38px;
        font-weight: 800;
        margin-top: 5px;
    }

    .info-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #171a21;
        border: 1px solid #30343b;
        min-height: 120px;
    }

    .info-title {
        font-size: 15px;
        color: #9ca3af;
    }

    .info-value {
        font-size: 25px;
        font-weight: 700;
        margin-top: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_tpr(group_data):
    """
    Calculate True Positive Rate for one group.
    """

    actual = group_data["actual_decision"]
    predicted = group_data["predicted_decision"]

    positive_actual = actual == 1

    total_positive = positive_actual.sum()

    if total_positive == 0:
        return 0.0

    true_positive = (
        predicted[positive_actual] == 1
    ).sum()

    return true_positive / total_positive


def calculate_fpr(group_data):
    """
    Calculate False Positive Rate for one group.
    """

    actual = group_data["actual_decision"]
    predicted = group_data["predicted_decision"]

    negative_actual = actual == 0

    total_negative = negative_actual.sum()

    if total_negative == 0:
        return 0.0

    false_positive = (
        predicted[negative_actual] == 1
    ).sum()

    return false_positive / total_negative


def get_risk_level(value):
    """
    Convert a fairness difference into a risk level.
    """

    value = abs(float(value))

    if value <= 0.05:
        return "LOW"

    elif value <= 0.10:
        return "MEDIUM"

    else:
        return "HIGH"


def calculate_overall_risk(
    dp_difference,
    eo_difference,
    tpr_difference,
    fpr_difference
):
    """
    Calculate overall ethical risk using the
    maximum fairness disparity.
    """

    differences = [
        abs(dp_difference),
        abs(eo_difference),
        abs(tpr_difference),
        abs(fpr_difference)
    ]

    maximum_difference = max(differences)

    if maximum_difference <= 0.05:
        return "LOW"

    elif maximum_difference <= 0.10:
        return "MEDIUM"

    else:
        return "HIGH"


# ============================================================
# LOAD AVAILABLE DATASETS
# ============================================================

if not os.path.exists(DATA_DIR):

    st.error(
        "Data folder was not found."
    )

    st.stop()


dataset_files = sorted(
    [
        file
        for file in os.listdir(DATA_DIR)
        if file.lower().endswith(".csv")
    ]
)


if not dataset_files:

    st.error(
        "No CSV datasets were found inside the data folder."
    )

    st.info(
        "Please add CSV datasets inside Ethical_AI_Framework/data/"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Dashboard Controls"
)


# ============================================================
# DATASET SELECTION
# ============================================================

st.sidebar.markdown(
    "### Dataset Selection"
)


selected_dataset_file = st.sidebar.selectbox(
    "Select dataset",
    dataset_files,
    index=(
        dataset_files.index("synthetic_decision_dataset.csv")
        if "synthetic_decision_dataset.csv" in dataset_files
        else 0
    )
)


DATASET_PATH = os.path.join(
    DATA_DIR,
    selected_dataset_file
)


# ============================================================
# LOAD SELECTED DATASET
# ============================================================

try:

    dataset = pd.read_csv(
        DATASET_PATH
    )

except Exception as e:

    st.error(
        f"Unable to load dataset: {e}"
    )

    st.stop()


# ============================================================
# CHECK DATASET COLUMNS
# ============================================================

required_columns = [
    "group",
    "decision"
]


missing_columns = [
    column
    for column in required_columns
    if column not in dataset.columns
]


if missing_columns:

    st.error(
        "The selected dataset is missing required columns: "
        + ", ".join(missing_columns)
    )

    st.info(
        "The dataset must contain at least: group and decision."
    )

    st.stop()


# ============================================================
# CLEAN DATASET
# ============================================================

dataset = dataset.copy()

dataset = dataset.dropna(
    subset=["group", "decision"]
)


# Convert decision to 0/1 if possible

try:

    dataset["decision"] = dataset["decision"].astype(int)

except Exception:

    unique_decisions = dataset["decision"].unique()

    if len(unique_decisions) == 2:

        decision_mapping = {
            unique_decisions[0]: 0,
            unique_decisions[1]: 1
        }

        dataset["decision"] = (
            dataset["decision"]
            .map(decision_mapping)
        )

    else:

        st.error(
            "The decision column must contain two classes."
        )

        st.stop()


# ============================================================
# GROUP SELECTION
# ============================================================

st.sidebar.markdown(
    "### Group Selection"
)


groups = sorted(
    dataset["group"]
    .dropna()
    .unique()
    .tolist()
)


selected_groups = st.sidebar.multiselect(
    "Select groups for analysis",
    groups,
    default=groups
)


st.sidebar.divider()


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.sidebar.markdown(
    "### Project Information"
)


st.sidebar.write(
    "**Model:** Random Forest"
)


st.sidebar.write(
    "**Fairness:** Fairlearn"
)


st.sidebar.write(
    f"**Dataset:** {selected_dataset_file}"
)


st.sidebar.write(
    "**Dashboard:** Streamlit"
)


st.sidebar.divider()


st.sidebar.info(
    "This dashboard is a prototype for identifying "
    "potential fairness and ethical risks in automated decisions."
)


# ============================================================
# CHECK GROUP SELECTION
# ============================================================

if not selected_groups:

    st.warning(
        "Please select at least one group from the sidebar."
    )

    st.stop()


# ============================================================
# SELECT DATA FOR ANALYSIS
# ============================================================

selected_dataset = dataset[
    dataset["group"].isin(
        selected_groups
    )
].copy()


if len(selected_dataset) == 0:

    st.warning(
        "No records found for the selected groups."
    )

    st.stop()


# ============================================================
# MODEL TRAINING FUNCTION
# ============================================================

@st.cache_data
def train_model_and_predict(
    data,
    dataset_name
):

    data = data.copy()

    target_column = "decision"

    # Do NOT use protected attribute "group"
    # as a model feature.
    feature_columns = [
        column
        for column in data.columns
        if column not in [
            target_column,
            "group"
        ]
    ]

    # If there are no other features,
    # use group only for prediction generation.
    if len(feature_columns) == 0:

        X = pd.DataFrame(
            {
                "constant_feature": [1] * len(data)
            }
        )

        feature_columns = [
            "constant_feature"
        ]

    else:

        X = data[
            feature_columns
        ].copy()

    y = data[
        target_column
    ].astype(int)

    groups_data = data[
        "group"
    ].copy()

    # --------------------------------------------------------
    # Handle categorical and numerical columns
    # --------------------------------------------------------

    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "numeric",
                "passthrough",
                numeric_features
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    try:

        X_train, X_test, y_train, y_test, groups_train, groups_test = (
            train_test_split(
                X,
                y,
                groups_data,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        )

    except ValueError:

        # Fallback when stratification is not possible

        X_train, X_test, y_train, y_test, groups_train, groups_test = (
            train_test_split(
                X,
                y,
                groups_data,
                test_size=0.20,
                random_state=42
            )
        )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = pipeline.predict(
        X_test
    )

    # --------------------------------------------------------
    # Prediction dataframe
    # --------------------------------------------------------

    predictions = pd.DataFrame(
        {
            "group": groups_test.values,
            "actual_decision": y_test.values,
            "predicted_decision": y_pred
        }
    )

    predictions = predictions.reset_index(
        drop=True
    )

    return (
        pipeline,
        predictions
    )


# ============================================================
# TRAIN MODEL
# ============================================================

try:

    model, predictions = train_model_and_predict(
        selected_dataset,
        selected_dataset_file
    )

except Exception as e:

    st.error(
        f"Model training failed: {e}"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '⚖️ Ethical AI Risk Dashboard'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="subtitle">'
    'Ethical AI Framework for Automated Decisions'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    """
    This system evaluates an automated decision model for
    **bias, fairness, and accountability risks** across different groups.
    """
)


st.divider()


# ============================================================
# SYSTEM OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📊 System Overview'
    '</div>',
    unsafe_allow_html=True
)


total_records = len(
    selected_dataset
)


total_approved = int(
    selected_dataset["decision"].sum()
)


total_rejected = (
    total_records -
    total_approved
)


if total_records > 0:

    approval_rate = (
        total_approved /
        total_records
    )

else:

    approval_rate = 0


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Decisions",
        f"{total_records:,}"
    )


with col2:

    st.metric(
        "Approved",
        f"{total_approved:,}"
    )


with col3:

    st.metric(
        "Rejected",
        f"{total_rejected:,}"
    )


with col4:

    st.metric(
        "Approval Rate",
        f"{approval_rate:.2%}"
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🤖 Model Performance'
    '</div>',
    unsafe_allow_html=True
)


y_true = predictions[
    "actual_decision"
]


y_pred = predictions[
    "predicted_decision"
]


accuracy = accuracy_score(
    y_true,
    y_pred
)


precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)


recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )


with col2:

    st.metric(
        "Precision",
        f"{precision:.2%}"
    )


with col3:

    st.metric(
        "Recall",
        f"{recall:.2%}"
    )


with col4:

    st.metric(
        "F1 Score",
        f"{f1:.2%}"
    )


st.divider()


# ============================================================
# GROUP-WISE ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '👥 Group-wise Decision Analysis'
    '</div>',
    unsafe_allow_html=True
)


filtered_predictions = predictions[
    predictions["group"].isin(
        selected_groups
    )
].copy()


# ============================================================
# GROUP APPROVAL RATES
# ============================================================

group_rates = (
    filtered_predictions
    .groupby("group")[
        "predicted_decision"
    ]
    .mean()
    .reset_index()
)


group_rates.columns = [
    "Group",
    "Approval Rate"
]


group_rates["Approval Rate"] *= 100


col1, col2 = st.columns(2)


# ============================================================
# APPROVAL RATE CHART
# ============================================================

with col1:

    st.subheader(
        "Approval Rate by Group"
    )

    fig = px.bar(
        group_rates,
        x="Group",
        y="Approval Rate",
        text="Approval Rate",
        title="Predicted Approval Rate"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="Approval Rate (%)",
        xaxis_title="Group"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# ============================================================
# DECISION DISTRIBUTION
# ============================================================

with col2:

    st.subheader(
        "Decision Distribution"
    )

    decision_distribution = (
        filtered_predictions[
            "predicted_decision"
        ]
        .value_counts()
        .reset_index()
    )

    decision_distribution.columns = [
        "Decision",
        "Count"
    ]

    decision_distribution[
        "Decision"
    ] = (
        decision_distribution[
            "Decision"
        ]
        .map(
            {
                0: "Rejected",
                1: "Approved"
            }
        )
    )

    fig2 = px.pie(
        decision_distribution,
        names="Decision",
        values="Count",
        hole=0.45,
        title="Predicted Decisions"
    )

    st.plotly_chart(
        fig2,
        width="stretch"
    )


st.divider()


# ============================================================
# FAIRNESS ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '⚖️ Fairness Analysis'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# FAIRNESS CALCULATION
# ============================================================

if (
    len(selected_groups) >= 2
    and len(filtered_predictions) > 0
):

    fairness_y_true = filtered_predictions[
        "actual_decision"
    ]

    fairness_y_pred = filtered_predictions[
        "predicted_decision"
    ]

    fairness_groups = filtered_predictions[
        "group"
    ]

    try:

        dp_difference = abs(
            demographic_parity_difference(
                y_true=fairness_y_true,
                y_pred=fairness_y_pred,
                sensitive_features=fairness_groups
            )
        )

    except Exception:

        dp_difference = 0.0


    try:

        eo_difference = abs(
            equalized_odds_difference(
                y_true=fairness_y_true,
                y_pred=fairness_y_pred,
                sensitive_features=fairness_groups
            )
        )

    except Exception:

        eo_difference = 0.0


    # --------------------------------------------------------
    # TPR AND FPR FOR EACH GROUP
    # --------------------------------------------------------

    group_fairness_results = []


    for group in selected_groups:

        group_data = filtered_predictions[
            filtered_predictions["group"] == group
        ].copy()

        if len(group_data) == 0:
            continue

        tpr = calculate_tpr(
            group_data
        )

        fpr = calculate_fpr(
            group_data
        )

        group_fairness_results.append(
            {
                "Group": group,
                "TPR": tpr,
                "FPR": fpr
            }
        )


    group_fairness_results = pd.DataFrame(
        group_fairness_results
    )


    if len(group_fairness_results) >= 2:

        tpr_difference = (
            group_fairness_results["TPR"].max()
            -
            group_fairness_results["TPR"].min()
        )

        fpr_difference = (
            group_fairness_results["FPR"].max()
            -
            group_fairness_results["FPR"].min()
        )

    else:

        tpr_difference = 0.0
        fpr_difference = 0.0


    fairness_available = True


else:

    dp_difference = None
    eo_difference = None
    tpr_difference = None
    fpr_difference = None

    fairness_available = False


# ============================================================
# FAIRNESS METRIC DISPLAY
# ============================================================

col1, col2, col3, col4 = st.columns(4)


if fairness_available:

    with col1:

        st.metric(
            "Demographic Parity",
            f"{dp_difference:.4f}"
        )


    with col2:

        st.metric(
            "Equalized Odds",
            f"{eo_difference:.4f}"
        )


    with col3:

        st.metric(
            "TPR Difference",
            f"{tpr_difference:.4f}"
        )


    with col4:

        st.metric(
            "FPR Difference",
            f"{fpr_difference:.4f}"
        )


else:

    with col1:

        st.metric(
            "Demographic Parity",
            "N/A"
        )


    with col2:

        st.metric(
            "Equalized Odds",
            "N/A"
        )


    with col3:

        st.metric(
            "TPR Difference",
            "N/A"
        )


    with col4:

        st.metric(
            "FPR Difference",
            "N/A"
        )


    if len(selected_groups) < 2:

        st.warning(
            "⚠️ Select at least two groups to calculate "
            "fairness differences."
        )


st.info(
    """
    **Interpretation:** Fairness differences closer to zero
    indicate smaller disparities between groups. These values
    are screening indicators and require contextual interpretation.
    """
)


# ============================================================
# FAIRNESS CHART
# ============================================================

if fairness_available:

    metric_chart = pd.DataFrame(
        {
            "Metric": [
                "Demographic Parity",
                "Equalized Odds",
                "TPR Difference",
                "FPR Difference"
            ],

            "Difference": [
                dp_difference,
                eo_difference,
                tpr_difference,
                fpr_difference
            ]
        }
    )


    fig3 = px.bar(
        metric_chart,
        x="Metric",
        y="Difference",
        text="Difference",
        title="Fairness Metric Comparison"
    )


    fig3.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )


    st.plotly_chart(
        fig3,
        width="stretch"
    )


else:

    st.info(
        "Fairness Metric Comparison requires at least "
        "two selected groups."
    )


st.divider()


# ============================================================
# ETHICAL RISK
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🚨 Ethical Risk Assessment'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DYNAMIC RISK CALCULATION
# ============================================================

if fairness_available:

    maximum_difference = max(
        abs(dp_difference),
        abs(eo_difference),
        abs(tpr_difference),
        abs(fpr_difference)
    )


    if maximum_difference <= 0.05:

        risk_level = "LOW"

        risk_class = "risk-low"

        risk_message = (
            "No large disparity was detected by the "
            "current screening criteria."
        )


    elif maximum_difference <= 0.10:

        risk_level = "MEDIUM"

        risk_class = "risk-medium"

        risk_message = (
            "A moderate fairness disparity was detected. "
            "Additional analysis is recommended."
        )


    else:

        risk_level = "HIGH"

        risk_class = "risk-high"

        risk_message = (
            "A substantial fairness disparity was detected. "
            "Further investigation is recommended."
        )


else:

    risk_level = "N/A"

    risk_class = "risk-medium"

    risk_message = (
        "Fairness risk cannot be calculated when only "
        "one group is selected. Select at least two groups."
    )


# ============================================================
# RISK CARD
# ============================================================

st.html(
    f"""
    <div class="risk-card {risk_class}">

        <div class="risk-title">
            OVERALL ETHICAL RISK
        </div>

        <div class="risk-value">
            {risk_level}
        </div>

        <div>
            {risk_message}
        </div>

    </div>
    """
)


# ============================================================
# RISK TABLE
# ============================================================

st.subheader(
    "Risk by Fairness Metric"
)


if fairness_available:

    display_risk = pd.DataFrame(
        {
            "Metric": [
                "Demographic Parity Difference",
                "Equalized Odds Difference",
                "TPR Difference",
                "FPR Difference"
            ],

            "Difference": [
                dp_difference,
                eo_difference,
                tpr_difference,
                fpr_difference
            ]
        }
    )


    display_risk["Risk Level"] = (
        display_risk["Difference"]
        .apply(get_risk_level)
    )


    display_risk[
        "Difference"
    ] = (
        display_risk[
            "Difference"
        ].round(4)
    )


    st.dataframe(
        display_risk,
        width="stretch",
        hide_index=True
    )


else:

    display_risk = pd.DataFrame(
        {
            "Metric": [
                "Demographic Parity Difference",
                "Equalized Odds Difference",
                "TPR Difference",
                "FPR Difference"
            ],

            "Difference": [
                "N/A",
                "N/A",
                "N/A",
                "N/A"
            ],

            "Risk Level": [
                "N/A",
                "N/A",
                "N/A",
                "N/A"
            ]
        }
    )


    st.dataframe(
        display_risk,
        width="stretch",
        hide_index=True
    )


st.divider()


# ============================================================
# BIAS FINDINGS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔍 Bias & Fairness Findings'
    '</div>',
    unsafe_allow_html=True
)


if fairness_available:

    # --------------------------------------------------------
    # DEMOGRAPHIC PARITY
    # --------------------------------------------------------

    if dp_difference >= 0.15:

        st.error(
            "⚠️ A substantial selection-rate disparity "
            "was detected between groups."
        )

    elif dp_difference >= 0.05:

        st.warning(
            "⚠️ A moderate selection-rate disparity "
            "was detected between groups."
        )

    else:

        st.success(
            "✓ Selection rates are relatively similar "
            "between groups."
        )


    # --------------------------------------------------------
    # TPR
    # --------------------------------------------------------

    if tpr_difference >= 0.05:

        st.warning(
            "A difference in True Positive Rates was detected."
        )

    else:

        st.success(
            "✓ True Positive Rates are relatively similar "
            "between groups."
        )


    # --------------------------------------------------------
    # FPR
    # --------------------------------------------------------

    if fpr_difference >= 0.05:

        st.warning(
            "A difference in False Positive Rates was detected."
        )

    else:

        st.success(
            "✓ False Positive Rates are relatively similar "
            "between groups."
        )


else:

    st.info(
        "Select at least two groups to generate "
        "bias and fairness findings."
    )


# ============================================================
# ACCOUNTABILITY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🛡️ Accountability & Recommendations'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DYNAMIC RECOMMENDATIONS
# ============================================================

if not fairness_available:

    recommendations = [

        "Select at least two groups for fairness comparison.",

        "Review the training data for representation and quality.",

        "Check whether features introduce indirect bias.",

        "Monitor fairness metrics after deployment.",

        "Document model decisions and fairness evaluations.",

        "Consider fairness-aware model development if disparities persist."

    ]

else:

    recommendations = []


    if dp_difference >= 0.05:

        recommendations.append(
            "Investigate the source of group-level "
            "selection-rate disparities."
        )


    if tpr_difference >= 0.05:

        recommendations.append(
            "Investigate differences in True Positive "
            "Rates between groups."
        )


    if fpr_difference >= 0.05:

        recommendations.append(
            "Investigate differences in False Positive "
            "Rates between groups."
        )


    recommendations.extend(
        [

            "Review the training data for representation and quality.",

            "Check whether features introduce indirect bias.",

            "Monitor fairness metrics after deployment.",

            "Document model decisions and fairness evaluations.",

            "Consider fairness-aware model development if disparities persist."

        ]
    )


for recommendation in recommendations:

    st.markdown(
        f"**•** {recommendation}"
    )


st.divider()


# ============================================================
# ETHICAL AI PIPELINE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔄 Ethical AI Evaluation Pipeline'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.info(
        "📁\n\n"
        "**DATA**\n\n"
        f"{selected_dataset_file}"
    )


with col2:

    st.info(
        "🤖\n\n"
        "**MODEL**\n\n"
        "Random Forest"
    )


with col3:

    st.info(
        "⚖️\n\n"
        "**FAIRNESS**\n\n"
        "Fairlearn"
    )


with col4:

    st.info(
        "🚨\n\n"
        "**RISK**\n\n"
        "Risk Analysis"
    )


with col5:

    st.info(
        "🛡️\n\n"
        "**ACCOUNTABILITY**\n\n"
        "Recommendations"
    )


st.divider()


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'ℹ️ Project Information'
    '</div>',
    unsafe_allow_html=True
)


info1, info2, info3 = st.columns(3)


# ============================================================
# DATASET CARD
# ============================================================

with info1:

    st.html(
        f"""
        <div class="info-card">

            <div class="info-title">
                Dataset
            </div>

            <div class="info-value">
                {len(dataset):,} Records
            </div>

        </div>
        """
    )


# ============================================================
# PROTECTED ATTRIBUTE CARD
# ============================================================

with info2:

    st.html(
        """
        <div class="info-card">

            <div class="info-title">
                Protected Attribute
            </div>

            <div class="info-value">
                Group
            </div>

        </div>
        """
    )


# ============================================================
# FAIRNESS FRAMEWORK CARD
# ============================================================

with info3:

    st.html(
        """
        <div class="info-card">

            <div class="info-title">
                Fairness Framework
            </div>

            <div class="info-value">
                Fairlearn
            </div>

        </div>
        """
    )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander(
    f"📄 View {selected_dataset_file}"
):

    st.dataframe(
        dataset,
        width="stretch",
        hide_index=True
    )


# ============================================================
# MODEL PREDICTIONS
# ============================================================

with st.expander(
    "🤖 View Model Predictions"
):

    st.dataframe(
        filtered_predictions,
        width="stretch",
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Ethical AI Framework for Automated Decisions | "
    "Python • Scikit-learn • Fairlearn • Streamlit"
)