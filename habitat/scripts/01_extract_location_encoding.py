"""
01_extract_location_encoding.py
Stage 1A: Extract Softmax probability vectors from 5-class location CSV as continuous encodings.
"""
import pandas as pd
import sys
sys.path.append(str(__import__('pathlib').Path(__file__).resolve().parent))
from config import (
    LOCATION_TRAIN_CSV, LOCATION_TEST_CSV, FEATURES_DIR,
    LOCATION_COLS_MAP, COL_ID
)


def extract_location_encoding(csv_path, output_path, split_name):
    print(f"\nExtracting continuous location encoding: {split_name}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples.")

    rename_map = {"id": COL_ID}
    rename_map.update(LOCATION_COLS_MAP)
    cols_to_keep = ["id"] + list(LOCATION_COLS_MAP.keys())
    df_out = df[cols_to_keep].rename(columns=rename_map)

    prob_cols = list(LOCATION_COLS_MAP.values())
    row_sums = df_out[prob_cols].sum(axis=1)
    max_deviation = (row_sums - 1.0).abs().max()
    print(f"Max row sum deviation: {max_deviation:.6f}")
    if max_deviation > 0.01:
        print("Warning: Probability sum deviates from 1.0 by more than 0.01.")
    else:
        print("Validation passed (row sum ≈ 1.0).")

    dominant = df_out[prob_cols].idxmax(axis=1).value_counts()
    print(f"\nDominant location distribution:")
    for loc, cnt in dominant.items():
        print(f"  {loc}: {cnt} ({cnt/len(df_out)*100:.1f}%)")

    df_out.to_csv(output_path, index=False)
    print(f"Saved to {output_path} ({len(df_out)} samples, {len(prob_cols)} features).")
    return df_out


def main():
    train_out = FEATURES_DIR / "location_encoding_train.csv"
    extract_location_encoding(LOCATION_TRAIN_CSV, train_out, "Train")

    test_out = FEATURES_DIR / "location_encoding_test.csv"
    extract_location_encoding(LOCATION_TEST_CSV, test_out, "Test")

    print("\nStage 1A Complete.")


if __name__ == "__main__":
    main()
