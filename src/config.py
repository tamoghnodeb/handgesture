"""Central configuration for GestureX.

Keeping paths, labels, and operational defaults in one place makes the
collection, experiment, benchmark, and deployment scripts agree on a single
dataset contract.  Paths are derived from this file, so commands work when run
from the repository root as documented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final


# Repository paths ---------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"

RAW_DATASET_PATH: Final[Path] = RAW_DATA_DIR / "gesture_landmarks.csv"
PROCESSED_DATASET_PATH: Final[Path] = PROCESSED_DATA_DIR / "gesture_features.csv"
# The validation-selected artifact used by deployment.  Per-representation
# experiment artifacts may live alongside it under more specific file names.
MODEL_ARTIFACT_PATH: Final[Path] = MODELS_DIR / "gesturex_best.joblib"
MODEL_METADATA_PATH: Final[Path] = MODELS_DIR / "gesturex_best_metadata.json"


# Dataset contract ---------------------------------------------------------
LANDMARK_COUNT: Final[int] = 21
LANDMARK_DIMENSIONS: Final[int] = 3
RAW_FEATURE_DIMENSION: Final[int] = LANDMARK_COUNT * LANDMARK_DIMENSIONS
INVARIANT_FEATURE_DIMENSION: Final[int] = 8

# Flattened raw landmarks always use this exact order:
# landmark_0_x, landmark_0_y, landmark_0_z, landmark_1_x, ... landmark_20_z.
RAW_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"landmark_{landmark_index}_{axis}"
    for landmark_index in range(LANDMARK_COUNT)
    for axis in ("x", "y", "z")
)

INVARIANT_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "thumb_tip_to_index_mcp_norm",
    "index_tip_to_index_mcp_norm",
    "middle_tip_to_middle_mcp_norm",
    "ring_tip_to_ring_mcp_norm",
    "pinky_tip_to_pinky_mcp_norm",
    "thumb_angle_deg",
    "index_finger_angle_deg",
    "middle_finger_angle_deg",
)

METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "session_id",
    "subject_id",
    "gesture",
    "timestamp",
)
DATASET_COLUMNS: Final[tuple[str, ...]] = METADATA_COLUMNS + RAW_FEATURE_COLUMNS


# Labels -------------------------------------------------------------------
# Canonical labels are compact, file-safe strings.  Display names are kept
# separately so UI wording never changes the training target representation.
GESTURE_LABELS: Final[tuple[str, ...]] = (
    "open_palm",
    "fist",
    "thumbs_up",
    "thumbs_down",
    "victory",
    "point_left",
    "point_right",
)
GESTURE_CLASSES: Final[tuple[str, ...]] = GESTURE_LABELS  # Backwards-friendly alias.
GESTURE_DISPLAY_NAMES: Final[dict[str, str]] = {
    "open_palm": "Open Palm",
    "fist": "Fist",
    "thumbs_up": "Thumbs Up",
    "thumbs_down": "Thumbs Down",
    "victory": "Victory",
    "point_left": "Point Left",
    "point_right": "Point Right",
}


# Reproducible model and evaluation defaults --------------------------------
RANDOM_SEED: Final[int] = 42
DEFAULT_VALIDATION_FRACTION: Final[float] = 0.20
DEFAULT_CV_FOLDS: Final[int] = 5

LOGISTIC_REGRESSION_PARAMS: Final[dict[str, object]] = {
    "max_iter": 2_000,
    "random_state": RANDOM_SEED,
}
RANDOM_FOREST_PARAMS: Final[dict[str, object]] = {
    "n_estimators": 300,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "class_weight": "balanced",
}
RBF_SVM_PARAMS: Final[dict[str, object]] = {
    "kernel": "rbf",
    "C": 3.0,
    "gamma": "scale",
    "probability": True,
    "random_state": RANDOM_SEED,
}


# MediaPipe, collection, and deployment defaults ---------------------------
DEFAULT_CAMERA_INDEX: Final[int] = 0
MAX_NUM_HANDS: Final[int] = 2
MIN_DETECTION_CONFIDENCE: Final[float] = 0.60
MIN_TRACKING_CONFIDENCE: Final[float] = 0.60
DEFAULT_SAMPLES_PER_GESTURE: Final[int] = 200
DEFAULT_COLLECTION_INTERVAL_MS: Final[int] = 100
DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.60
DEFAULT_SMOOTHING_WINDOW: Final[int] = 7
DEFAULT_BENCHMARK_REPETITIONS: Final[int] = 100

# Feature extraction constants ---------------------------------------------
NORMALIZATION_EPSILON: Final[float] = 1e-6
RAW_REPRESENTATION: Final[str] = "raw"
INVARIANT_REPRESENTATION: Final[str] = "invariant"
SUPPORTED_REPRESENTATIONS: Final[tuple[str, ...]] = (
    RAW_REPRESENTATION,
    INVARIANT_REPRESENTATION,
)


def canonicalize_gesture_label(label: object) -> str:
    """Return a canonical GestureX label or raise a useful ``ValueError``.

    Collection is friendlier when users can type ``"Open Palm"`` or
    ``"open-palm"``, while the saved CSV still has only one stable spelling.
    """

    if not isinstance(label, str):
        raise ValueError(f"Gesture label must be a string, received {label!r}.")

    cleaned = label.strip().lower().replace("-", "_").replace(" ", "_")
    if cleaned not in GESTURE_LABELS:
        allowed = ", ".join(GESTURE_LABELS)
        raise ValueError(
            f"Unsupported gesture label {label!r}. Allowed labels: {allowed}."
        )
    return cleaned


def feature_columns(representation: str) -> tuple[str, ...]:
    """Return ordered feature names for ``raw`` or ``invariant`` data."""

    normalized = representation.strip().lower()
    if normalized == RAW_REPRESENTATION:
        return RAW_FEATURE_COLUMNS
    if normalized == INVARIANT_REPRESENTATION:
        return INVARIANT_FEATURE_COLUMNS
    supported = ", ".join(SUPPORTED_REPRESENTATIONS)
    raise ValueError(
        f"Unsupported feature representation {representation!r}. "
        f"Choose one of: {supported}."
    )


def ensure_project_directories() -> None:
    """Create runtime data/output directories if they do not yet exist."""

    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
