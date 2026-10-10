import pandas as pd


# ============================================================
# ETHICAL AI - RISK ANALYSIS
# ============================================================

print("=" * 65)
print("ETHICAL AI - ETHICAL RISK ANALYSIS")
print("=" * 65)


# ============================================================
# 1. LOAD FAIRNESS RESULTS
# ============================================================

summary_path = "data/fairness_summary.csv"
metrics_path = "data/fairness_metrics.csv"

summary = pd.read_csv(summary_path)
metrics = pd.read_csv(metrics_path)


# ============================================================
# 2. EXTRACT FAIRNESS METRICS
# ============================================================

def get_metric(metric_name):
    row = summary[summary["metric"] == metric_name]

    if len(row) == 0:
        return 0.0

    return float(row["value"].iloc[0])


dp_difference = get_metric(
    "demographic_parity_difference"
)

eo_difference = get_metric(
    "equalized_odds_difference"
)

tpr_difference = get_metric(
    "tpr_difference"
)

fpr_difference = get_metric(
    "fpr_difference"
)


# ============================================================
# 3. RISK LEVEL FUNCTION
# ============================================================

def calculate_risk(value):
    """
    Project-defined screening thresholds.

    < 0.05  -> LOW
    < 0.15  -> MEDIUM
    >= 0.15 -> HIGH
    """

    if value < 0.05:
        return "LOW"

    elif value < 0.15:
        return "MEDIUM"

    else:
        return "HIGH"


# ============================================================
# 4. CALCULATE INDIVIDUAL RISKS
# ============================================================

dp_risk = calculate_risk(dp_difference)

eo_risk = calculate_risk(eo_difference)

tpr_risk = calculate_risk(tpr_difference)

fpr_risk = calculate_risk(fpr_difference)


# ============================================================
# 5. CONVERT RISK TO NUMERICAL SCORE
# ============================================================

risk_scores = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3
}

dp_score = risk_scores[dp_risk]
eo_score = risk_scores[eo_risk]
tpr_score = risk_scores[tpr_risk]
fpr_score = risk_scores[fpr_risk]


# ============================================================
# 6. OVERALL RISK
# ============================================================

overall_score = max(
    dp_score,
    eo_score,
    tpr_score,
    fpr_score
)

if overall_score == 1:
    overall_risk = "LOW"

elif overall_score == 2:
    overall_risk = "MEDIUM"

else:
    overall_risk = "HIGH"


# ============================================================
# 7. DISPLAY FAIRNESS METRICS
# ============================================================

print("\nFAIRNESS METRICS")
print("-" * 65)

print(
    f"Demographic Parity Difference : "
    f"{dp_difference:.4f}"
)

print(
    f"Equalized Odds Difference    : "
    f"{eo_difference:.4f}"
)

print(
    f"TPR Difference               : "
    f"{tpr_difference:.4f}"
)

print(
    f"FPR Difference               : "
    f"{fpr_difference:.4f}"
)


# ============================================================
# 8. DISPLAY RISK LEVELS
# ============================================================

print("\nRISK BY METRIC")
print("-" * 65)

print(f"Demographic Parity : {dp_risk}")
print(f"Equalized Odds    : {eo_risk}")
print(f"TPR               : {tpr_risk}")
print(f"FPR               : {fpr_risk}")


# ============================================================
# 9. OVERALL RISK
# ============================================================

print("\n" + "=" * 65)
print("OVERALL ETHICAL RISK")
print("=" * 65)

print(f"\nOverall Risk Level: {overall_risk}")


# ============================================================
# 10. GENERATE FINDINGS
# ============================================================

findings = []

if dp_difference >= 0.15:
    findings.append(
        "A substantial difference in selection rates "
        "was detected between groups."
    )

elif dp_difference >= 0.05:
    findings.append(
        "A moderate difference in selection rates "
        "was detected between groups."
    )

else:
    findings.append(
        "Selection rates are relatively similar "
        "between groups."
    )


if tpr_difference >= 0.15:
    findings.append(
        "A substantial difference in true positive "
        "rates was detected."
    )

elif tpr_difference >= 0.05:
    findings.append(
        "A moderate difference in true positive "
        "rates was detected."
    )


if fpr_difference >= 0.15:
    findings.append(
        "A substantial difference in false positive "
        "rates was detected."
    )

elif fpr_difference >= 0.05:
    findings.append(
        "A moderate difference in false positive "
        "rates was detected."
    )


# ============================================================
# 11. DISPLAY FINDINGS
# ============================================================

print("\nKEY FINDINGS")
print("-" * 65)

for i, finding in enumerate(findings, start=1):
    print(f"{i}. {finding}")


# ============================================================
# 12. RECOMMENDATIONS
# ============================================================

recommendations = [
    "Investigate the source of group-level disparities.",
    "Review the training data for representation and quality.",
    "Evaluate whether relevant features introduce indirect bias.",
    "Monitor fairness metrics regularly after deployment.",
    "Consider fairness-aware model development if disparities persist."
]

print("\nRECOMMENDATIONS")
print("-" * 65)

for i, recommendation in enumerate(recommendations, start=1):
    print(f"{i}. {recommendation}")


# ============================================================
# 13. SAVE RISK REPORT
# ============================================================

risk_report = pd.DataFrame({
    "metric": [
        "Demographic Parity Difference",
        "Equalized Odds Difference",
        "TPR Difference",
        "FPR Difference"
    ],
    "value": [
        dp_difference,
        eo_difference,
        tpr_difference,
        fpr_difference
    ],
    "risk_level": [
        dp_risk,
        eo_risk,
        tpr_risk,
        fpr_risk
    ]
})

risk_report.to_csv(
    "data/ethical_risk_report.csv",
    index=False
)


# ============================================================
# 14. SAVE OVERALL RISK
# ============================================================

overall_report = pd.DataFrame({
    "overall_risk": [overall_risk],
    "overall_score": [overall_score]
})

overall_report.to_csv(
    "data/overall_risk.csv",
    index=False
)


print("\n✓ Risk report saved to:")
print("data/ethical_risk_report.csv")

print("\n✓ Overall risk saved to:")
print("data/overall_risk.csv")

print("\n" + "=" * 65)
print("ETHICAL RISK ANALYSIS COMPLETED")
print("=" * 65)