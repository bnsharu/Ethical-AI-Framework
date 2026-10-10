import pandas as pd
import os

# -----------------------------
# Load dataset
# -----------------------------

input_path = "data/synthetic_decision_dataset.csv"

df = pd.read_csv(input_path)

print("=" * 60)
print("DATA PREPROCESSING AND VALIDATION")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")

# -----------------------------
# 1. Check missing values
# -----------------------------

print("\n1. Missing Values")
print("-" * 30)

missing_values = df.isnull().sum()

print(missing_values)

if missing_values.sum() == 0:
    print("\n✓ No missing values found.")
else:
    print("\n⚠ Missing values detected.")

# -----------------------------
# 2. Check duplicate records
# -----------------------------

print("\n2. Duplicate Records")
print("-" * 30)

duplicates = df.duplicated().sum()

print(f"Number of duplicate rows: {duplicates}")

if duplicates == 0:
    print("✓ No duplicate records found.")
else:
    print("⚠ Duplicate records detected.")

# -----------------------------
# 3. Check data types
# -----------------------------

print("\n3. Data Types")
print("-" * 30)

print(df.dtypes)

# -----------------------------
# 4. Validate numerical columns
# -----------------------------

print("\n4. Numerical Data Summary")
print("-" * 30)

numerical_columns = [
    "age",
    "income",
    "credit_score",
    "employment_years",
    "loan_amount",
    "previous_default"
]

print(df[numerical_columns].describe())

# -----------------------------
# 5. Validate categorical columns
# -----------------------------

print("\n5. Categorical Columns")
print("-" * 30)

categorical_columns = [
    "education",
    "gender",
    "group"
]

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts())

# -----------------------------
# 6. Validate target variable
# -----------------------------

print("\n6. Target Variable")
print("-" * 30)

print(df["decision"].value_counts())

print("\nDecision distribution:")
print(df["decision"].value_counts(normalize=True))

# -----------------------------
# 7. Validate protected attribute
# -----------------------------

print("\n7. Protected Attribute")
print("-" * 30)

print("Protected attribute: group")

print(df["group"].value_counts())

# -----------------------------
# 8. Check invalid values
# -----------------------------

print("\n8. Data Validation")
print("-" * 30)

invalid_age = ((df["age"] < 18) | (df["age"] > 100)).sum()

invalid_income = (df["income"] < 0).sum()

invalid_credit = (
    (df["credit_score"] < 300) |
    (df["credit_score"] > 850)
).sum()

invalid_decision = (~df["decision"].isin([0, 1])).sum()

print(f"Invalid age values: {invalid_age}")
print(f"Invalid income values: {invalid_income}")
print(f"Invalid credit scores: {invalid_credit}")
print(f"Invalid decision values: {invalid_decision}")

# -----------------------------
# Final validation
# -----------------------------

print("\n" + "=" * 60)

if (
    missing_values.sum() == 0
    and duplicates == 0
    and invalid_age == 0
    and invalid_income == 0
    and invalid_credit == 0
    and invalid_decision == 0
):
    print("✓ DATA VALIDATION PASSED")
else:
    print("⚠ DATA VALIDATION REQUIRES ATTENTION")

print("=" * 60)