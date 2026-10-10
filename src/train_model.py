import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. LOAD DATASET
# ============================================================

input_path = "data/synthetic_decision_dataset.csv"

df = pd.read_csv(input_path)

print("=" * 60)
print("ETHICAL AI - ML DECISION MODEL")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

# Decision is what the model will predict
X = df.drop("decision", axis=1)
y = df["decision"]


# ============================================================
# 3. REMOVE PROTECTED ATTRIBUTE FROM MODEL FEATURES
# ============================================================

# We do NOT use 'group' directly to make the decision.
# We keep it separately for fairness analysis.

sensitive_features = X["group"].copy()

X = X.drop("group", axis=1)


# ============================================================
# 4. IDENTIFY COLUMN TYPES
# ============================================================

categorical_features = [
    "education",
    "gender"
]

numerical_features = [
    "age",
    "income",
    "credit_score",
    "employment_years",
    "loan_amount",
    "previous_default"
]


# ============================================================
# 5. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 6. CREATE ML MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)


# ============================================================
# 7. CREATE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
    X,
    y,
    sensitive_features,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("✓ Model training completed.")


# ============================================================
# 10. MAKE PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_test)


# ============================================================
# 11. MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 12. CREATE RESULTS DATAFRAME
# ============================================================

results = X_test.copy()

results["actual_decision"] = y_test.values
results["predicted_decision"] = y_pred
results["group"] = group_test.values


# ============================================================
# 13. GROUP-WISE PREDICTION RATE
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION RATE BY GROUP")
print("=" * 60)

group_prediction_rate = (
    results.groupby("group")["predicted_decision"]
    .mean()
)

print(group_prediction_rate)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

os.makedirs("data", exist_ok=True)

results.to_csv(
    "data/model_predictions.csv",
    index=False
)

# Save trained model
joblib.dump(
    pipeline,
    "data/ethical_ai_model.pkl"
)

print("\n✓ Predictions saved to:")
print("data/model_predictions.csv")

print("\n✓ Trained model saved to:")
print("data/ethical_ai_model.pkl")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)