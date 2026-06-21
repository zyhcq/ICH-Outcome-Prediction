"""
03_extract_pash_features.py
Stage 1C: Extract PASH (Perihematomal Adaptive Spatial Heterogeneity) features.
"""
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from pathlib import Path
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    IMAGES_TRAIN_DIR, IMAGES_TEST_DIR,
    MASKS_TRAIN_DIR, MASKS_TEST_DIR,
    FEATURES_DIR, COL_ID, PASH_COLS,
)


def extract_pash_features_single(ct_array, mask_array):
    mask_bin = (mask_array > 0).astype(np.uint8)
    n_voxels = int(np.sum(mask_bin))

    if n_voxels < 100:
        return {
            "PASH_high_compactness": 0.0,
            "PASH_low_dispersion": 0.0,
            "PASH_fragmentation": 1,
        }

    hu_values = ct_array[mask_bin > 0].astype(float)

    q_low = np.percentile(hu_values, 33)
    q_high = np.percentile(hu_values, 66)

    high_density = (ct_array > q_high) & (mask_bin > 0)
    low_density = (ct_array < q_low) & (mask_bin > 0)

    n_high = int(np.sum(high_density))
    if n_high > 0:
        labeled_high, num_high = ndimage.label(high_density)
        component_sizes = ndimage.sum(high_density, labeled_high, range(1, num_high + 1))
        max_component_size = float(np.max(component_sizes))
        pash_high_comp = max_component_size / n_high
    else:
        pash_high_comp = 0.0

    n_low = int(np.sum(low_density))
    if n_low > 10:
        low_coords = np.argwhere(low_density)
        centroid = low_coords.mean(axis=0)
        dists = np.sqrt(np.sum((low_coords - centroid) ** 2, axis=1))
        dispersion = float(np.std(dists))

        equiv_radius = (3 * n_voxels / (4 * np.pi)) ** (1 / 3)
        pash_low_disp = dispersion / equiv_radius if equiv_radius > 0 else 0.0
    else:
        pash_low_disp = 0.0

    if n_high > 0:
        _, pash_frag = ndimage.label(high_density)
    else:
        pash_frag = 1

    return {
        "PASH_high_compactness": round(pash_high_comp, 6),
        "PASH_low_dispersion": round(pash_low_disp, 6),
        "PASH_fragmentation": int(pash_frag),
    }


def extract_for_split(images_dir, masks_dir, output_path, split_name):
    print(f"\nExtracting PASH features: {split_name}")

    mask_files = sorted(list(Path(masks_dir).glob("*.nii.gz")))
    print(f"Found {len(mask_files)} mask files.")

    existing_ids = set()
    if Path(output_path).exists():
        df_existing = pd.read_csv(output_path)
        existing_ids = set(df_existing[COL_ID].values)
        print(f"Found existing results for {len(existing_ids)} cases, skipping them.")

    results = []
    errors = []

    for mask_path in tqdm(mask_files, desc=f"{split_name} PASH features"):
        patient_id = mask_path.name.replace(".nii.gz", "")

        if patient_id in existing_ids:
            continue

        ct_path = Path(images_dir) / f"{patient_id}_0000.nii.gz"
        if not ct_path.exists():
            errors.append(f"CT file missing: {ct_path}")
            continue

        try:
            ct_img = nib.load(str(ct_path))
            ct_array = ct_img.get_fdata()
            mask_img = nib.load(str(mask_path))
            mask_array = mask_img.get_fdata()

            feats = extract_pash_features_single(ct_array, mask_array)
            feats[COL_ID] = patient_id
            results.append(feats)
        except Exception as e:
            errors.append(f"{patient_id}: {str(e)}")

    if results:
        df_new = pd.DataFrame(results)
        if existing_ids and Path(output_path).exists():
            df_existing = pd.read_csv(output_path)
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = df_new

        cols = [COL_ID] + PASH_COLS
        df_all = df_all[cols]
        df_all.to_csv(output_path, index=False)
        print(f"Saved to {output_path} ({len(df_all)} cases, {len(PASH_COLS)} features)")
    else:
        print("Warning: No new results extracted.")

    if errors:
        print(f"\n{len(errors)} Errors:")
        for e in errors[:10]:
            print(f"  {e}")


def main():
    extract_for_split(
        IMAGES_TRAIN_DIR, MASKS_TRAIN_DIR,
        FEATURES_DIR / "pash_features_train.csv", "Train"
    )
    extract_for_split(
        IMAGES_TEST_DIR, MASKS_TEST_DIR,
        FEATURES_DIR / "pash_features_test.csv", "Test"
    )

    print("\nStage 1C Complete.")


if __name__ == "__main__":
    main()
