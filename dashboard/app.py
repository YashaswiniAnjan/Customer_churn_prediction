from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path
import sys


# ============================================================
# 1. Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"


# ============================================================
# 2. Import retention strategy module
# ============================================================

sys.path.insert(0, str(BASE_DIR))

from src.retention_strategy import get_retention_strategy


# ============================================================
# 3. Create Flask application
# ============================================================

app = Flask(__name__)


# ============================================================
# 4. Load trained XGBoost model
# ============================================================

model = joblib.load(
    MODEL_DIR / "xgboost.pkl"
)


# ============================================================
# 5. Load feature columns used during training
# ============================================================

feature_columns = joblib.load(
    MODEL_DIR / "feature_columns.pkl"
)


# ============================================================
# 6. Dashboard home page
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# 7. Churn prediction
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------------------------
    # Get customer information from HTML form
    # --------------------------------------------------------

    tenure = float(
        request.form["tenure"]
    )

    monthly_charges = float(
        request.form["monthly_charges"]
    )

    total_charges = float(
        request.form["total_charges"]
    )

    contract = request.form["contract"]

    internet_service = request.form["internet_service"]

    dependents = request.form["dependents"]

    online_security = request.form["online_security"]

    payment_method = request.form["payment_method"]

    paperless_billing = request.form["paperless_billing"]


    # --------------------------------------------------------
    # Create dataframe using EXACTLY the 10 model features
    # --------------------------------------------------------

    input_data = pd.DataFrame([{

        "Tenure Months": tenure,

        "Internet Service_Fiber optic":
            1 if internet_service == "Fiber optic" else 0,

        "Monthly Charges": monthly_charges,

        "Contract_Two year":
            1 if contract == "Two year" else 0,

        "Contract_One year":
            1 if contract == "One year" else 0,

        "Dependents_Yes":
            1 if dependents == "Yes" else 0,

        "Total Charges": total_charges,

        "Payment Method_Electronic check":
            1 if payment_method == "Electronic check" else 0,

        "Online Security_Yes":
            1 if online_security == "Yes" else 0,

        "Paperless Billing_Yes":
            1 if paperless_billing == "Yes" else 0

    }])


    # --------------------------------------------------------
    # Match exact feature order from training
    # --------------------------------------------------------

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # --------------------------------------------------------
    # Make churn prediction
    # --------------------------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    # --------------------------------------------------------
    # Get churn probability
    # --------------------------------------------------------

    probability = model.predict_proba(
        input_data
    )[0][1]


    # --------------------------------------------------------
    # Convert prediction into readable result
    # --------------------------------------------------------

    if prediction == 1:

        result = "Customer likely to churn"

    else:

        result = "Customer likely to stay"


    # Convert probability to percentage

    probability_percentage = round(
        probability * 100,
        2
    )


    # --------------------------------------------------------
    # Generate retention strategies
    # --------------------------------------------------------

    retention_strategies = get_retention_strategy(

        churn_prediction=prediction,

        churn_probability=probability,

        tenure=tenure,

        monthly_charges=monthly_charges,

        contract=contract,

        internet_service=internet_service,

        dependents=dependents,

        online_security=online_security,

        payment_method=payment_method,

        paperless_billing=paperless_billing
    )


    # --------------------------------------------------------
    # Display prediction and retention strategy
    # --------------------------------------------------------

    return render_template(

        "index.html",

        result=result,

        probability=probability_percentage,

        retention_strategies=retention_strategies
    )


# ============================================================
# 8. Run Flask application
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )