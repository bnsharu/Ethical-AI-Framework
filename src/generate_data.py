import pandas as pd
import numpy as np

# Reproducibility
np.random.seed(42)

# Number of records
n = 5000

# -----------------------------
# Generate applicant information
# -----------------------------

age = np.random.randint(21, 61, n)

income = np.random.randint(20000, 150000, n)

credit_score = np.random.randint(450, 850, n)

employment_years = np.random.randint(0, 21, n)

education = np.random.choice(
    ["High School", "Bachelor", "Master", "PhD"],
    n,
    p=[0.30, 0.40, 0.25, 0.05]
)

gender = np.random.choice(
    ["Male", "Female"],
    n,
    p=[0.55, 0.45]
)

# Protected/sensitive group
group = np.random.choice(
    ["Group_A", "Group_B"],
    n,
    p=[0.60, 0.40]
)

loan_amount = np.random.randint(5000, 100000, n)

previous_default = np.random.choice(
    [0, 1],
    n,
    p=[0.85, 0.15]
)

# -----------------------------
# Generate decision score
# -----------------------------

score = (
    0.35 * (credit_score - 450) / 400
    + 0.25 * (income - 20000) / 130000
    + 0.15 * employment_years / 20
    + 0.10 * age / 60
    - 0.20 * previous_default
)

# Add controlled group disparity
# Group B receives a lower decision score
score = np.where(
    group == "Group_B",
    score - 0.12,
    score
)

# Add random variation
score += np.random.normal(0, 0.08, n)

# Convert score into automated decision
decision = (score > 0.45).astype(int)

# -----------------------------
# Create DataFrame
# -----------------------------

df = pd.DataFrame({
    "age": age,
    "income": income,
    "credit_score": credit_score,
    "employment_years": employment_years,
    "education": education,
    "gender": gender,
    "group": group,
    "loan_amount": loan_amount,
    "previous_default": previous_default,
    "decision": decision
})

# -----------------------------
# Save dataset
# -----------------------------

output_path = "data/synthetic_decision_dataset.csv"

df.to_csv(output_path, index=False)

print("Dataset generated successfully!")
print(f"Number of records: {len(df)}")
print(f"Dataset saved to: {output_path}")

print("\nFirst 5 records:")
print(df.head())

print("\nGroup distribution:")
print(df["group"].value_counts())

print("\nOverall decision distribution:")
print(df["decision"].value_counts())

print("\nDecision rate by group:")
print(df.groupby("group")["decision"].mean())