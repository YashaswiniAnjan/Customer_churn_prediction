import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Telco_customer_churn.xlsx"
PROCESSED_PATH = BASE_DIR / "data" / "processed_churn_data.csv"


# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------

print("Loading customer churn dataset...")

df = pd.read_excel(DATA_PATH)

print(f"Original dataset shape: {df.shape}")


# ---------------------------------------------------------
# 3. Remove duplicate records
# ---------------------------------------------------------

duplicates = df.duplicated().sum()

print(f"Duplicate records found: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()


# ---------------------------------------------------------
# 4. Convert Total Charges to numeric
# ---------------------------------------------------------

if "Total Charges" in df.columns:
    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"],
        errors="coerce"
    )


# ---------------------------------------------------------
# 5. Create target variable
# ---------------------------------------------------------

df["Churn"] = df["Churn Label"].map({
    "Yes": 1,
    "No": 0
})


# ---------------------------------------------------------
# 6. Remove columns that should not be used for prediction
# ---------------------------------------------------------

columns_to_drop = [
    "CustomerID",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason",
    "Churn Category",
    "Customer Status"
]

columns_to_drop = [
    col for col in columns_to_drop
    if col in df.columns
]

df = df.drop(columns=columns_to_drop)


# ---------------------------------------------------------
# 7. Handle any rows affected by numeric conversion
# ---------------------------------------------------------

df = df.dropna()


# ---------------------------------------------------------
# 8. Convert categorical variables using One-Hot Encoding
# ---------------------------------------------------------

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)


# ---------------------------------------------------------
# 9. Convert boolean columns to integers
# ---------------------------------------------------------

boolean_columns = df.select_dtypes(
    include=["bool"]
).columns

df[boolean_columns] = df[boolean_columns].astype(int)


# ---------------------------------------------------------
# 10. Save processed dataset
# ---------------------------------------------------------

df.to_csv(PROCESSED_PATH, index=False)


# ---------------------------------------------------------
# 11. Display preprocessing results
# ---------------------------------------------------------

print("\nPreprocessing completed successfully!")
print(f"Processed dataset shape: {df.shape}")
print(f"Processed dataset saved to: {PROCESSED_PATH}")

print("\nTarget distribution:")
print(df["Churn"].value_counts())

print("\nFirst 5 rows:")
print(df.head())