"""Simple data registration / validation script."""
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tourism.csv"

EXPECTED_COLS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]

def main():
    print("=" * 50)
    print("DATA REGISTRATION / VALIDATION")
    print("=" * 50)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"File size: {DATA_PATH.stat().st_size / 1024:.1f} KB")

    missing = set(EXPECTED_COLS) - set(df.columns)
    extra   = set(df.columns) - set(EXPECTED_COLS)
    if missing:
        print("\nMissing expected columns:", missing)
    else:
        print("\nAll expected columns present")
    if extra:
        print("Extra columns (will be ignored later):", extra)

    print("\n--- Target distribution ---")
    print(df["ProdTaken"].value_counts(normalize=True).round(3))

    print("\n--- Null counts (should be 0) ---")
    print(df.isnull().sum().sum(), "total nulls")

    print("\n--- Quick numeric summary ---")
    print(df[["Age", "MonthlyIncome", "NumberOfTrips"]].describe().round(1))

    print("\nRegistration complete.")
    return df

if __name__ == "__main__":
    main()
