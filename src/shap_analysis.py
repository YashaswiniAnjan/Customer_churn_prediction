import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

from pathlib import Path


# ---------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# 2. Load test data
# ---------------------------------------------------------

print("Loading test data...")

test_data = pd.read_csv(
    REPORT_DIR / "test_predictions_data.csv"
)

X_test = test_data.drop(
    "Actual_Churn",
    axis=1
)

y_test = test_data["Actual_Churn"]

# Use a smaller sample for SHAP analysis
X_test = X_test.sample(
    n=min(200, len(X_test)),
    random_state=42
)

y_test = y_test.loc[X_test.index]

print("SHAP sample shape:", X_test.shape)

print("Test data shape:", X_test.shape)


# ---------------------------------------------------------
# 3. Load XGBoost model
# ---------------------------------------------------------

print("Loading XGBoost model...")

model_path = MODEL_DIR / "xgboost.pkl"

if not model_path.exists():
    raise FileNotFoundError(
        "xgboost.pkl was not found in the models folder."
    )

model = joblib.load(model_path)


# ---------------------------------------------------------
# 4. Create SHAP explainer
# ---------------------------------------------------------

print("Creating SHAP explainer...")

explainer = shap.Explainer(
    model.predict,
    X_test
)

shap_values = explainer(
    X_test
)


# ---------------------------------------------------------
# 5. SHAP Summary Plot
# ---------------------------------------------------------

print("Generating SHAP summary plot...")

plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values.values,
    X_test,
    show=False
)

plt.title(
    "SHAP Feature Importance - XGBoost Churn Model"
)

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 6. SHAP Bar Plot
# ---------------------------------------------------------

print("Generating SHAP feature importance chart...")

plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values.values,
    X_test,
    plot_type="bar",
    show=False
)

plt.title(
    "Mean Absolute SHAP Feature Importance"
)

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 7. Calculate feature importance
# ---------------------------------------------------------

print("Calculating feature importance...")

importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean_Absolute_SHAP": abs(shap_values.values).mean(axis=0)
})

importance = importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

importance.to_csv(
    REPORT_DIR / "shap_feature_importance.csv",
    index=False
)


# ---------------------------------------------------------
# 8. Display top features
# ---------------------------------------------------------

print("\nTop 10 features influencing churn:")

print(
    importance.head(10).to_string(index=False)
)


# ---------------------------------------------------------
# 9. Completion message
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nSHAP files saved in:")

print(REPORT_DIR)