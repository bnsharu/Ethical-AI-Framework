import subprocess
import sys


def run_step(description, command):
    print("\n" + "=" * 65)
    print(description)
    print("=" * 65)

    result = subprocess.run(
        [sys.executable] + command,
        capture_output=False
    )

    if result.returncode != 0:
        print(f"\nERROR: {description} failed.")
        sys.exit(1)

    print(f"\n✓ {description} completed successfully.")


# ============================================================
# STEP 1 - GENERATE DATA
# ============================================================

run_step(
    "STEP 1: GENERATING SYNTHETIC DATASET",
    ["src/generate_data.py"]
)


# ============================================================
# STEP 2 - PREPROCESS AND VALIDATE
# ============================================================

run_step(
    "STEP 2: VALIDATING DATASET",
    ["src/preprocess.py"]
)


# ============================================================
# STEP 3 - TRAIN MODEL
# ============================================================

run_step(
    "STEP 3: TRAINING ML MODEL",
    ["src/train_model.py"]
)


# ============================================================
# STEP 4 - FAIRNESS ANALYSIS
# ============================================================

run_step(
    "STEP 4: PERFORMING FAIRNESS ANALYSIS",
    ["src/fairness_analysis.py"]
)


# ============================================================
# STEP 5 - ETHICAL RISK ANALYSIS
# ============================================================

run_step(
    "STEP 5: CALCULATING ETHICAL RISK",
    ["src/risk_analysis.py"]
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("COMPLETE ETHICAL AI PIPELINE FINISHED")
print("=" * 65)

print("""
Generated outputs:

1. Synthetic dataset
2. Validated dataset
3. Trained ML model
4. Model predictions
5. Fairness metrics
6. Ethical risk report
7. Overall risk assessment

To launch the dashboard:

    streamlit run dashboard/app.py
""")