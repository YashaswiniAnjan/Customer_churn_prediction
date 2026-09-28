# ============================================================
# Customer Churn Prediction - Retention Strategy Module
# ============================================================


def get_retention_strategy(
    churn_prediction,
    churn_probability,
    tenure,
    monthly_charges,
    contract,
    internet_service,
    dependents,
    online_security,
    payment_method,
    paperless_billing
):
    """
    Generate rule-based retention recommendations
    based on customer information and churn prediction.
    """

    strategies = []

    # --------------------------------------------------------
    # Only recommend retention actions for churn-risk customers
    # --------------------------------------------------------

    if churn_prediction != 1:
        return [
            "No immediate retention action required.",
            "Continue regular customer engagement and service monitoring."
        ]


    # --------------------------------------------------------
    # High churn probability
    # --------------------------------------------------------

    if churn_probability >= 0.70:
        strategies.append(
            "Prioritize this customer for proactive retention outreach."
        )


    # --------------------------------------------------------
    # Contract-related strategy
    # --------------------------------------------------------

    if contract == "Month-to-month":
        strategies.append(
            "Offer a suitable longer-term contract option with an "
            "appropriate customer incentive."
        )


    # --------------------------------------------------------
    # Monthly charges strategy
    # --------------------------------------------------------

    if monthly_charges >= 70:
        strategies.append(
            "Review the customer's current plan and available "
            "lower-cost or value-added options."
        )


    # --------------------------------------------------------
    # Internet service strategy
    # --------------------------------------------------------

    if internet_service == "Fiber optic":
        strategies.append(
            "Review fiber-optic service satisfaction and provide "
            "support or plan options if appropriate."
        )


    # --------------------------------------------------------
    # Online security strategy
    # --------------------------------------------------------

    if online_security == "No":
        strategies.append(
            "Offer an online security option or trial to improve "
            "the customer's service value."
        )


    # --------------------------------------------------------
    # Payment method strategy
    # --------------------------------------------------------

    if payment_method == "Electronic check":
        strategies.append(
            "Offer assistance with payment options and review "
            "whether an alternative payment method is more convenient."
        )


    # --------------------------------------------------------
    # New customer strategy
    # --------------------------------------------------------

    if tenure <= 12:
        strategies.append(
            "Provide early-stage customer support and engagement "
            "to strengthen customer retention."
        )


    # --------------------------------------------------------
    # Dependents-related engagement
    # --------------------------------------------------------

    if dependents == "No":
        strategies.append(
            "Review available individual-customer plans or benefits "
            "that may better match the customer's needs."
        )


    # --------------------------------------------------------
    # Paperless billing
    # --------------------------------------------------------

    if paperless_billing == "Yes":
        strategies.append(
            "Review billing clarity and ensure the customer "
            "understands current charges and plan details."
        )


    # --------------------------------------------------------
    # Fallback strategy
    # --------------------------------------------------------

    if not strategies:
        strategies.append(
            "Contact the customer to understand service concerns "
            "and provide a personalized retention offer."
        )


    return strategies