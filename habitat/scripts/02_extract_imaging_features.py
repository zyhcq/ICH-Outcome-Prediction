"""
02_extract_imaging_features.py
Stage 1B: Extract volume/morphology/density statistical features (14 features) from CT + Mask
"""
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import skew, kurtosis
from pathlib import Path
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    IMAGES_TRAIN_DIR, IMAGES_TEST_DIR,
    MASKS_TRAIN_DIR, MASKS_TEST_DIR,
    FEATURES_DIR, COL_ID, IMAGING_ALL_COLS,
)


def load_nifti(path):
    img = nib.load(str(path))
    data = img.get_fdata()
    spacing = img.header.get_zooms()[:3]
    return data, spacing


def extract_imaging_features_single(ct_array, mask_array, voxel_spacing):
    features = {}
    voxel_vol = float(np.prod(voxel_spacing))
    mask_bin = (mask_array > 0).astype(np.uint8)
    n_voxels = int(np.sum(mask_bin))

    if n_voxels == 0:
        return {col: 0.0 for col in IMAGING_ALL_COLS}

    volume_mm3 = n_voxels * voxel_vol
    volume_ml = volume_mm3 / 1000.0

    features["hematoma_volume_ml"] = volume_ml
    features["hematoma_volume_log"] = np.log(volume_ml + 1)

    intracranial_mask = ct_array > -100
    intracranial_vol_mm3 = float(np.sum(intracranial_mask)) * voxel_vol
    intracranial_vol_ml = intracranial_vol_mm3 / 1000.0
    features["relative_volume"] = volume_ml / intracranial_vol_ml if intracranial_vol_ml > 0 else 0

    try:
        from skimage.measure import marching_cubes, mesh_surface_area
        verts, faces, _, _ = marching_cubes(mask_bin, level=0.5, spacing=voxel_spacing)
        sa = mesh_surface_area(verts, faces)
        features["surface_area"] = sa
        features["sphericity"] = (36 * np.pi * volume_mm3**2)**(1/3) / sa if sa > 0 else 0
        features["surface_volume_ratio"] = sa / volume_mm3 if volume_mm3 > 0 else 0
    except Exception:
        features["surface_area"] = 0.0
        features["sphericity"] = 0.0
        features["surface_volume_ratio"] = 0.0

    coords = np.argwhere(mask_bin > 0)
    bbox_size = (coords.max(axis=0) - coords.min(axis=0) + 1).astype(float) * np.array(voxel_spacing)
    bbox_vol = float(np.prod(bbox_size))
    features["compactness"] = volume_mm3 / bbox_vol if bbox_vol > 0 else 0

    coords_mm = coords.astype(float) * np.array(voxel_spacing)
    centered = coords_mm - coords_mm.mean(axis=0)
    try:
        cov = np.cov(centered.T)
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
        features["elongation"] = np.sqrt(eigenvalues[1] / eigenvalues[0]) if eigenvalues[0] > 0 else 1.0
    except Exception:
        features["elongation"] = 1.0

    hu_values = ct_array[mask_bin > 0].astype(float)
    features["density_mean"] = float(np.mean(hu_values))
    features["density_std"] = float(np.std(hu_values))
    features["density_skewness"] = float(skew(hu_values)) if len(hu_values) > 2 else 0.0
    features["density_kurtosis"] = float(kurtosis(hu_values)) if len(hu_values) > 2 else 0.0
    features["density_max"] = float(np.max(hu_values))
    features["density_range"] = float(np.max(hu_values) - np.min(hu_values))

    return features


def extract_for_split(images_dir, masks_dir, output_path, split_name):
    print(f"\nExtracting imaging features: {split_name}")

    mask_files = sorted(list(Path(masks_dir).glob("*.nii.gz")))
    print(f"Found {len(mask_files)} mask files.")

    existing_ids = set()
    if Path(output_path).exists():
        df_existing = pd.read_csv(output_path)
        existing_ids = set(df_existing[COL_ID].values)
        print(f"Found existing results for {len(existing_ids)} cases, skipping them.")

    results = []
    errors = []

    for mask_path in tqdm(mask_files, desc=f"{split_name} features"):
        patient_id = mask_path.name.replace(".nii.gz", "")

        if patient_id in existing_ids:
            continue

        ct_path = Path(images_dir) / f"{patient_id}_0000.nii.gz"
        if not ct_path.exists():
            errors.append(f"CT file missing: {ct_path}")
            continue

        try:
            ct_array, spacing = load_nifti(ct_path)
            mask_array, _ = load_nifti(mask_path)

            mask_array = (mask_array > 0).astype(np.uint8)

            feats = extract_imaging_features_single(ct_array, mask_array, spacing)
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

        cols = [COL_ID] + IMAGING_ALL_COLS
        df_all = df_all[cols]
        df_all.to_csv(output_path, index=False)
        print(f"Saved to {output_path} ({len(df_all)} cases, {len(IMAGING_ALL_COLS)} features)")
    else:
        print("Warning: No new results extracted.")

    if errors:
        print(f"\n{len(errors)} Errors:")
        for e in errors[:10]:
            print(f"  {e}")

    return len(results) + len(existing_ids)


def main():
    extract_for_split(
        IMAGES_TRAIN_DIR, MASKS_TRAIN_DIR,
        FEATURES_DIR / "imaging_features_train.csv", "Train"
    )

    extract_for_split(
        IMAGES_TEST_DIR, MASKS_TEST_DIR,
        FEATURES_DIR / "imaging_features_test.csv", "Test"
    )

    print("\nStage 1B Complete.")


if __name__ == "__main__":
    main()
