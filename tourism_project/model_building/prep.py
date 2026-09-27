"""Data preparation script."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_PATH = DATA_DIR / "tourism.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH  = DATA_DIR / "test.csv"

DROP_COLS = ["CustomerID", "Unnamed: 0"]

def clean(df):
    df = df.copy()
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    # fix the known Gender typo
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    for col in ["NumberOfFollowups", "PreferredPropertyStar",
                "NumberOfTrips", "NumberOfChildrenVisiting"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median()).astype(int)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def main():
    print("Loading raw data …")
    df = pd.read_csv(RAW_PATH)
    print("Raw shape:", df.shape)
    df = clean(df)
    print("After cleaning:", df.shape)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["ProdTaken"]
    )
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"Train saved → {TRAIN_PATH}  ({len(train_df)} rows)")
    print(f"Test  saved → {TEST_PATH}   ({len(test_df)} rows)")
    print("Target ratio (train):", train_df["ProdTaken"].mean().round(3))
    print("Prep done.")

if __name__ == "__main__":
    main()
