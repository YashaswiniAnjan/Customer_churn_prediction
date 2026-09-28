import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)


# ---------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# 2. Load test data and model comparison
# ---------------------------------------------------------

print("Loading evaluation data...")

test_data = pd.read_csv(
    REPORT_DIR / "test_predictions_data.csv"
)

comparison = pd.read_csv(
    REPORT_DIR / "model_comparison.csv"
)

X_test = test_data.drop("Actual_Churn", axis=1)
y_test = test_data["Actual_Churn"]


print("Test data shape:", X_test.shape)


# ---------------------------------------------------------
# 3. Load scaler
# ---------------------------------------------------------

scaler = joblib.load(
    MODEL_DIR / "scaler.pkl"
)


# ---------------------------------------------------------
# 4. Model names and files
# ---------------------------------------------------------

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "K-Nearest Neighbors": "k_nearest_neighbors.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "XGBoost": "xgboost.pkl"
}


scaled_models = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Support Vector Machine"
]


# ---------------------------------------------------------
# 5. ROC Curve
# ---------------------------------------------------------

print("\nGenerating ROC curve...")

plt.figure(figsize=(10, 7))

for name, filename in model_files.items():

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        print(f"Skipping {name}: model file not found.")
        continue

    model = joblib.load(model_path)

    if name in scaled_models:
        X_input = scaler.transform(X_test)
    else:
        X_input = X_test

    probability = model.predict_proba(X_input)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probability
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curves - Customer Churn Prediction")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "roc_curves.png",
    dpi=300
)

plt.close()

print("ROC curve saved successfully.")


# ---------------------------------------------------------
# 6. Confusion Matrix for every model
# ---------------------------------------------------------

print("\nGenerating confusion matrices...")

for name, filename in model_files.items():

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        continue

    model = joblib.load(model_path)

    if name in scaled_models:
        X_input = scaler.transform(X_test)
    else:
        X_input = X_test

    predictions = model.predict(X_input)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"]
    )

    display.plot()

    plt.title(
        f"Confusion Matrix - {name}"
    )

    plt.tight_layout()

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    plt.savefig(
        REPORT_DIR / f"confusion_matrix_{safe_name}.png",
        dpi=300
    )

    plt.close()


print("Confusion matrices saved successfully.")


# ---------------------------------------------------------
# 7. Model comparison chart
# ---------------------------------------------------------

print("\nGenerating model comparison chart...")

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

comparison_plot = comparison.set_index("Model")[metrics]

comparison_plot.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.xlabel("Machine Learning Model")
plt.ylabel("Score")

plt.title(
    "Performance Comparison of Machine Learning Models"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.ylim(0, 1)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "model_performance_comparison.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 8. Save final evaluation summary
# ---------------------------------------------------------

comparison.to_csv(
    REPORT_DIR / "final_model_evaluation.csv",
    index=False
)


print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nEvaluation files saved in:")
print(REPORT_DIR)