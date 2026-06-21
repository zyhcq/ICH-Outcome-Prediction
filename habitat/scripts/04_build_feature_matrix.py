"""
04_build_feature_matrix.py
Combine all extracted features to build the final feature matrix.
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    FEATURES_DIR, CLINICAL_TRAIN_CSV, CLINICAL_TEST_CSV,
    COL_ID, COL_SURGERY, COL_PRIMARY_ENDPOINT,
    CLINICAL_COLS_MAP, LOCATION_LABEL_COL, 
    LOCATION_COLS, IMAGING_ALL_COLS, PASH_COLS,
    CLINICAL_FEATURE_COLS,
)


def load_clinical(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns=CLINICAL_COLS_MAP)
    keep_cols = [COL_ID, COL_SURGERY, LOCATION_LABEL_COL, COL_PRIMARY_ENDPOINT]
    keep_cols += list(CLINICAL_COLS_MAP.values())
    df = df[keep_cols]
    return df


def build_for_split(clinical_csv, split_name, suffix):
    print(f"\nBuilding feature matrix: {split_name}")

    df_clinical = load_clinical(clinical_csv)
    n_total = len(df_clinical)
    print(f"Clinical data: {n_total} cases")

    n_surgery = int(df_clinical[COL_SURGERY].sum())
    df_clinical = df_clinical[df_clinical[COL_SURGERY] == 0].copy()
    n_after = len(df_clinical)
    print(f"Excluded surgery patients: {n_surgery} -> Remaining: {n_after}")

    df_loc = pd.read_csv(FEATURES_DIR / f"location_encoding_{suffix}.csv")
    df_img = pd.read_csv(FEATURES_DIR / f"imaging_features_{suffix}.csv")
    df_pash = pd.read_csv(FEATURES_DIR / f"pash_features_{suffix}.csv")
    print(f"Location encoding: {len(df_loc)} cases")
    print(f"Imaging features: {len(df_img)} cases")
    print(f"PASH features: {len(df_pash)} cases")

    df = df_clinical.merge(df_loc, on=COL_ID, how="left")
    df = df.merge(df_img, on=COL_ID, how="left")
    df = df.merge(df_pash, on=COL_ID, how="left")

    n_missing_loc = df[LOCATION_COLS[0]].isna().sum()
    n_missing_img = df[IMAGING_ALL_COLS[0]].isna().sum() if IMAGING_ALL_COLS[0] in df.columns else -1
    n_missing_pash = df[PASH_COLS[0]].isna().sum() if PASH_COLS[0] in df.columns else -1
    
    print("\nMissing values after merge:")
    print(f"  Location encoding missing: {n_missing_loc}")
    print(f"  Imaging features missing: {n_missing_img}")
    print(f"  PASH features missing: {n_missing_pash}")

    feature_cols = LOCATION_COLS + IMAGING_ALL_COLS + PASH_COLS + CLINICAL_FEATURE_COLS
    for col in feature_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            n_fill = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            print(f"  Imputed {col}: {n_fill} missing values -> median {median_val:.4f}")

    all_feature_cols = LOCATION_COLS + IMAGING_ALL_COLS + PASH_COLS + CLINICAL_FEATURE_COLS
    meta_cols = [COL_ID, LOCATION_LABEL_COL, COL_PRIMARY_ENDPOINT]
    output_cols = meta_cols + all_feature_cols

    df_out = df[output_cols]

    print("\nFinal feature matrix statistics:")
    print(f"Samples: {len(df_out)}")
    print(f"Features: {len(all_feature_cols)} (Location: {len(LOCATION_COLS)}, Imaging: {len(IMAGING_ALL_COLS)}, "
          f"PASH: {len(PASH_COLS)}, Clinical: {len(CLINICAL_FEATURE_COLS)})")

    y_primary = df_out[COL_PRIMARY_ENDPOINT]
    print(f"\nPrimary Endpoint ({COL_PRIMARY_ENDPOINT}):")
    print(f"  Poor outcome (1): {int(y_primary.sum())} ({y_primary.mean()*100:.1f}%)")
    print(f"  Good outcome (0): {int((1-y_primary).sum())} ({(1-y_primary.mean())*100:.1f}%)")

    fully_missing = [c for c in all_feature_cols if df_out[c].isna().all()]
    if fully_missing:
        print(f"\nWarning: Completely missing columns: {fully_missing}")
    else:
        print("\nValidation passed: No completely missing columns.")

    remaining_na = df_out[all_feature_cols].isna().sum().sum()
    print(f"Remaining total missing values: {remaining_na}")

    return df_out, output_cols


def main():
    df_train, cols = build_for_split(CLINICAL_TRAIN_CSV, "Train", "train")
    train_out = FEATURES_DIR / "feature_matrix_train.csv"
    df_train.to_csv(train_out, index=False)
    print(f"Train matrix saved to {train_out}")

    df_test, _ = build_for_split(CLINICAL_TEST_CSV, "Test", "test")
    test_out = FEATURES_DIR / "feature_matrix_test.csv"
    df_test.to_csv(test_out, index=False)
    print(f"Test matrix saved to {test_out}")

    print("\nFeature extraction pipeline complete.")
    print(f"  Train: {len(df_train)} cases")
    print(f"  Test: {len(df_test)} cases")
    print(f"  Features: {len(cols) - 3}")


if __name__ == "__main__":
    main()
