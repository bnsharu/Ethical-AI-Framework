# Ethical AI Framework for Automated Decisions

## 1. Project Overview

This project develops an Ethical AI framework for evaluating automated
decision systems for potential bias, fairness and accountability risks.

A synthetic automated loan-decision dataset is generated and used to
train a Random Forest classification model.

Fairlearn is then used to evaluate fairness across different groups.

The results are converted into an ethical-risk assessment and presented
through an interactive Streamlit dashboard.

---

## 2. Objectives

The main objectives are:

1. Generate a synthetic automated-decision dataset.
2. Train a machine-learning decision model.
3. Evaluate model performance.
4. Measure fairness across different groups.
5. Identify potential bias and fairness disparities.
6. Classify the observed disparities into risk levels.
7. Provide accountability recommendations.
8. Present the results using an interactive dashboard.

---

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Fairlearn
- Plotly
- Streamlit
- Joblib

---

## 4. System Workflow

Synthetic Dataset
        ↓
Data Validation
        ↓
Machine Learning Model
        ↓
Model Predictions
        ↓
Fairness Analysis
        ↓
Ethical Risk Assessment
        ↓
Streamlit Dashboard

---

## 5. Dataset

The project uses a synthetic dataset containing 5,000 records.

Features include:

- Age
- Income
- Credit Score
- Employment Years
- Education
- Gender
- Group
- Loan Amount
- Previous Default
- Decision

The `group` attribute is retained separately for fairness auditing.

The `decision` variable represents the automated decision:

- 1 = Approved
- 0 = Rejected

---

## 6. Machine Learning Model

A Random Forest Classifier is used to model the automated decision process.

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

---

## 7. Fairness Analysis

Fairlearn is used to calculate fairness metrics including:

- Selection Rate
- Demographic Parity Difference
- True Positive Rate
- False Positive Rate
- Equalized Odds Difference

These metrics help identify differences in model outcomes across groups.

---

## 8. Ethical Risk Assessment

The prototype uses project-defined screening thresholds.

| Difference | Risk |
|---|---|
| < 0.05 | Low |
| 0.05 - < 0.15 | Medium |
| >= 0.15 | High |

These thresholds are used only for project-level screening and are not
intended to represent universal legal or regulatory standards.

---

## 9. Dashboard

The Streamlit dashboard displays:

- System overview
- Model performance
- Group-wise decision analysis
- Fairness metrics
- Ethical risk level
- Bias findings
- Accountability recommendations
- Dataset and prediction information

---

## 10. Running the Project

### Activate virtual environment

Windows:

```bash
venv\Scripts\activate