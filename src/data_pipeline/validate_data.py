import pandas as pd
from pathlib import Path


def validate_data(file_path="data/synthetic_supply_chain.csv"):
    df = pd.read_csv(file_path)

    print("=== DATA VALIDATION ===")

    # Check columns
    required_columns = [
        "date",
        "demand",
        "disruption",
        "lead_time_days"
    ]

    assert all(col in df.columns for col in required_columns)
    print("✓ Required columns present")

    # Check missing values
    assert df.isnull().sum().sum() == 0
    print("✓ No missing values")

    # Demand must be non-negative
    assert (df["demand"] >= 0).all()
    print("✓ Demand values valid")

    # Disruption must be 0 or 1
    assert df["disruption"].isin([0, 1]).all()
    print("✓ Disruption values valid")

    # Lead time must be positive
    assert (df["lead_time_days"] > 0).all()
    print("✓ Lead time values valid")

    print("\n=== DATASET SUMMARY ===")
    print(f"Records: {len(df)}")
    print(f"Average demand: {df['demand'].mean():.2f}")
    print(f"Disruption rate: {df['disruption'].mean():.2%}")
    print(f"Average lead time: {df['lead_time_days'].mean():.2f} days")

    print("\n✓ DATA VALIDATION PASSED")


if __name__ == "__main__":
    validate_data()