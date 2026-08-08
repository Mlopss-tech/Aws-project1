import pandas as pd
import great_expectations as ge
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    """

    print("🔍 Starting data validation with Great Expectations...")

    # ------------------------------------------------------------------
    # Make a copy so original dataframe isn't modified
    # ------------------------------------------------------------------
    df = df.copy()

    # ------------------------------------------------------------------
    # Convert numeric columns BEFORE validation
    # ------------------------------------------------------------------
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = (
            df["TotalCharges"]
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
        )

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(
            df["MonthlyCharges"],
            errors="coerce"
        )

    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(
            df["tenure"],
            errors="coerce"
        )

    # ------------------------------------------------------------------
    # Create Great Expectations dataset
    # ------------------------------------------------------------------
    ge_df = ge.dataset.PandasDataset(df)

    print("📋 Validating schema and required columns...")

    # Required columns
    required_columns = [
        "customerID",
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    for col in required_columns:
        ge_df.expect_column_to_exist(col)

    ge_df.expect_column_values_to_not_be_null("customerID")

    print("💼 Validating business logic constraints...")

    ge_df.expect_column_values_to_be_in_set(
        "gender",
        ["Male", "Female"]
    )

    ge_df.expect_column_values_to_be_in_set(
        "Partner",
        ["Yes", "No"]
    )

    ge_df.expect_column_values_to_be_in_set(
        "Dependents",
        ["Yes", "No"]
    )

    ge_df.expect_column_values_to_be_in_set(
        "PhoneService",
        ["Yes", "No"]
    )

    ge_df.expect_column_values_to_be_in_set(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year",
        ]
    )

    ge_df.expect_column_values_to_be_in_set(
        "InternetService",
        [
            "DSL",
            "Fiber optic",
            "No",
        ]
    )

    print("📊 Validating numeric ranges...")

    ge_df.expect_column_values_to_be_between(
        "tenure",
        min_value=0,
        max_value=120
    )

    ge_df.expect_column_values_to_be_between(
        "MonthlyCharges",
        min_value=0,
        max_value=200
    )

    # Allow a few nulls because blank TotalCharges exist in Telco dataset
    ge_df.expect_column_values_to_be_between(
        "TotalCharges",
        min_value=0,
        mostly=0.99
    )

    print("📈 Validating missing values...")

    ge_df.expect_column_values_to_not_be_null(
        "tenure"
    )

    ge_df.expect_column_values_to_not_be_null(
        "MonthlyCharges"
    )

    ge_df.expect_column_values_to_not_be_null(
        "TotalCharges",
        mostly=0.99
    )

    print("🔗 Validating data consistency...")

    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95,
    )

    print("⚙️ Running validation suite...")

    results = ge_df.validate()

    failed_expectations = []

    for r in results["results"]:
        if not r["success"]:
            failed_expectations.append(
                r["expectation_config"]["expectation_type"]
            )

    total = len(results["results"])
    passed = sum(r["success"] for r in results["results"])

    print(f"✅ Passed: {passed}/{total}")

    if failed_expectations:
        print("\n❌ Failed Expectations:")
        for e in failed_expectations:
            print(f"   - {e}")

    return results["success"], failed_expectations