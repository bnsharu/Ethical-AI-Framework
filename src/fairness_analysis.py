import pandas as pd

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    demographic_parity_difference,
    equalized_odds_difference
)

# ============================================================
# 1. LOAD MODEL PREDICTIONS
# ============================================================

input_path = "data/model_predictions.csv"

df = pd.read_csv(input_path)

print("=" * 65)
print("ETHICAL AI - FAIRNESS ANALYSIS")
print("=" * 65)

print(f"\nNumber of test records: {len(df)}")


# ============================================================
# 2. DEFINE VARIABLES
# ============================================================

y_true = df["actual_decision"]

y_pred = df["predicted_decision"]

sensitive_features = df["group"]


# ============================================================
# 3. GROUP-WISE METRICS
# ============================================================

print("\n" + "=" * 65)
print("GROUP-WISE FAIRNESS METRICS")
print("=" * 65)

metric_frame = MetricFrame(
    metrics={
        "Selection Rate": selection_rate,
        "True Positive Rate": true_positive_rate,
        "False Positive Rate": false_positive_rate
    },
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)

print("\nMetrics by group:\n")

print(metric_frame.by_group)


# ============================================================
# 4. SELECTION RATE
# ============================================================

print("\n" + "=" * 65)
print("SELECTION / APPROVAL RATE")
print("=" * 65)

selection_rates = metric_frame.by_group["Selection Rate"]

print(selection_rates)

print("\nInterpretation:")
for group, rate in selection_rates.items():
    print(f"{group}: {rate:.2%} predicted approval rate")


# ============================================================
# 5. DEMOGRAPHIC PARITY DIFFERENCE
# ============================================================

dp_difference = demographic_parity_difference(
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)

print("\n" + "=" * 65)
print("DEMOGRAPHIC PARITY")
print("=" * 65)

print(f"\nDemographic Parity Difference: {dp_difference:.4f}")

print(
    "\nThis measures the difference in selection rates "
    "between groups."
)


# ============================================================
# 6. EQUALIZED ODDS DIFFERENCE
# ============================================================

eo_difference = equalized_odds_difference(
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)

print("\n" + "=" * 65)
print("EQUALIZED ODDS")
print("=" * 65)

print(f"\nEqualized Odds Difference: {eo_difference:.4f}")

print(
    "\nThis measures the largest difference in "
    "true-positive and false-positive rates between groups."
)


# ============================================================
# 7. TRUE POSITIVE RATE DIFFERENCE
# ============================================================

tpr = metric_frame.by_group["True Positive Rate"]

tpr_difference = tpr.max() - tpr.min()

print("\n" + "=" * 65)
print("TRUE POSITIVE RATE")
print("=" * 65)

print(tpr)

print(f"\nTPR Difference: {tpr_difference:.4f}")


# ============================================================
# 8. FALSE POSITIVE RATE DIFFERENCE
# ============================================================

fpr = metric_frame.by_group["False Positive Rate"]

fpr_difference = fpr.max() - fpr.min()

print("\n" + "=" * 65)
print("FALSE POSITIVE RATE")
print("=" * 65)

print(fpr)

print(f"\nFPR Difference: {fpr_difference:.4f}")


# ============================================================
# 9. FAIRNESS RISK SCREENING
# ============================================================

print("\n" + "=" * 65)
print("FAIRNESS RISK SCREENING")
print("=" * 65)

# Project-defined screening thresholds.
# These are NOT universal legal thresholds.

if dp_difference < 0.05:
    risk = "LOW"
elif dp_difference < 0.15:
    risk = "MEDIUM"
else:
    risk = "HIGH"

print(f"\nEthical Risk Level: {risk}")

if risk == "LOW":
    print("No large selection-rate disparity detected.")

elif risk == "MEDIUM":
    print("A moderate selection-rate disparity was detected.")
    print("Further investigation is recommended.")

else:
    print("A substantial selection-rate disparity was detected.")
    print("Further investigation is strongly recommended.")


# ============================================================
# 10. SAVE FAIRNESS RESULTS
# ============================================================

fairness_results = metric_frame.by_group.copy()

fairness_results.to_csv(
    "data/fairness_metrics.csv"
)

summary = pd.DataFrame({
    "metric": [
        "demographic_parity_difference",
        "equalized_odds_difference",
        "tpr_difference",
        "fpr_difference"
    ],
    "value": [
        dp_difference,
        eo_difference,
        tpr_difference,
        fpr_difference
    ]
})

summary.to_csv(
    "data/fairness_summary.csv",
    index=False
)

print("\n✓ Group-wise metrics saved to:")
print("data/fairness_metrics.csv")

print("\n✓ Fairness summary saved to:")
print("data/fairness_summary.csv")

print("\n" + "=" * 65)
print("FAIRNESS ANALYSIS COMPLETED")
print("=" * 65)