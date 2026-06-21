"""
Global Configuration
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

SEG_DATA_DIR = ROOT_DIR / "data"
IMAGES_TRAIN_DIR = SEG_DATA_DIR / ""
IMAGES_TEST_DIR = SEG_DATA_DIR / ""
MASKS_TRAIN_DIR = SEG_DATA_DIR / ""
MASKS_TEST_DIR = SEG_DATA_DIR / ""
CLINICAL_TRAIN_CSV = SEG_DATA_DIR / ""
CLINICAL_TEST_CSV = SEG_DATA_DIR / ""

LOCATION_DIR = ROOT_DIR / "location"
LOCATION_FEATURE_CSV = SEG_DATA_DIR / ""
LOCATION_TRAIN_CSV = LOCATION_DIR / ""
LOCATION_TEST_CSV = LOCATION_DIR / ""

FEATURES_DIR = ROOT_DIR / "features"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

for d in [FEATURES_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

COL_ID = "New_ID"
COL_SURGERY = "Surgery"
COL_PRIMARY_ENDPOINT = "90_day poor outcome 3_6"

CLINICAL_COLS_MAP = {
    "Age": "age",
    "Admission GCS score": "GCS",
    "IVH on first CT": "IVH",
}

LOCATION_COLS = ["p_basal", "p_brainstem", "p_cerebellum", "p_lobar", "p_thalamus"]

LOCATION_COLS_MAP = {
    "Prob_Basal": "p_basal",
    "Prob_Brainstem": "p_brainstem",
    "Prob_Cerebellum": "p_cerebellum",
    "Prob_Lobar": "p_lobar",
    "Prob_Thalamus": "p_thalamus",
}

LOCATION_LABEL_COL = "ICH location"
LOCATION_LABEL_MAP = {0: "basal", 1: "brainstem", 2: "cerebellum", 3: "lobar", 4: "thalamus"}

IMAGING_VOLUME_COLS = ["hematoma_volume_ml", "hematoma_volume_log", "relative_volume"]
IMAGING_MORPHOLOGY_COLS = ["sphericity", "surface_area", "compactness", "elongation", "surface_volume_ratio"]
IMAGING_DENSITY_COLS = ["density_mean", "density_std", "density_skewness", "density_kurtosis", "density_max", "density_range"]
IMAGING_ALL_COLS = IMAGING_VOLUME_COLS + IMAGING_MORPHOLOGY_COLS + IMAGING_DENSITY_COLS

PASH_COLS = ["PASH_high_compactness", "PASH_low_dispersion", "PASH_fragmentation"]

CLINICAL_FEATURE_COLS = list(CLINICAL_COLS_MAP.values())
