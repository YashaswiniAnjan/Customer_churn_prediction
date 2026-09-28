import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Telco_customer_churn.xlsx"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# 2. Load original dataset
# ---------------------------------------------------------

print("Loading dataset for Exploratory Data Analysis...")

df = pd.read_excel(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ---------------------------------------------------------
# 3. Basic dataset information
# ---------------------------------------------------------

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe(include="all").T)


# ---------------------------------------------------------
# 4. Churn distribution
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn Label"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "churn_distribution.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 5. Churn by Contract Type
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn Label"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "churn_by_contract.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 6. Churn by Internet Service
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Internet Service",
    hue="Churn Label"
)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "churn_by_internet_service.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 7. Churn by Payment Method
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="Payment Method",
    hue="Churn Label"
)

plt.title("Customer Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")

plt.xticks(rotation=30, ha="right")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "churn_by_payment_method.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 8. Monthly Charges vs Churn
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn Label",
    y="Monthly Charges"
)

plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "monthly_charges_vs_churn.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 9. Tenure vs Churn
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn Label",
    y="Tenure Months"
)

plt.title("Tenure vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "tenure_vs_churn.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 10. Churn percentage
# ---------------------------------------------------------

churn_percentage = (
    df["Churn Label"]
    .value_counts(normalize=True)
    * 100
)

print("\nChurn Percentage:")
print(churn_percentage)


# ---------------------------------------------------------
# 11. Save churn summary
# ---------------------------------------------------------

churn_summary = (
    df.groupby("Churn Label")
    .size()
    .reset_index(name="Customer Count")
)

churn_summary["Percentage"] = (
    churn_summary["Customer Count"]
    / churn_summary["Customer Count"].sum()
    * 100
)

churn_summary.to_csv(
    REPORTS_DIR / "churn_summary.csv",
    index=False
)


print("\nEDA completed successfully!")

print(f"EDA reports saved to: {REPORTS_DIR}")