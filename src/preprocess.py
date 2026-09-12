"""Dataset validation and leakage-aware feature preparation for GestureX.

This module validates the on-disk landmark CSV before model code sees it.  It
does *not* fit a scaler or otherwise learn from samples: fitting preprocessing
on all sessions would leak held-out cross-session information into training.
Model-specific scaling belongs inside a scikit-learn Pipeline fitted only on a
training partition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:  # Supports direct scripts and package imports.
    from .config import (
        DATASET_COLUMNS,
        GESTURE_LABELS,
        METADATA_COLUMNS,
        RAW_FEATURE_COLUMNS,
        RAW_REPRESENTATION,
        canonicalize_gesture_label,
    )
    from .features import (
        FeatureExtractionError,
        feature_names,
        normalize_representation,
        transform_feature_frame,
    )
except ImportError:  # pragma: no cover - exercised by direct script execution.
    from config import (  # type: ignore
        DATASET_COLUMNS,
        GESTURE_LABELS,
        METADATA_COLUMNS,
        RAW_FEATURE_COLUMNS,
        RAW_REPRESENTATION,
        canonicalize_gesture_label,
    )
    from features import (  # type: ignore
        FeatureExtractionError,
        feature_names,
        normalize_representation,
        transform_feature_frame,
    )


class DatasetValidationError(ValueError):
    """Raised when a landmark CSV does not meet the GestureX data contract."""


def _require_dataframe(dataframe: Any) -> pd.DataFrame:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    return dataframe


def _missing_or_blank(series: pd.Series) -> pd.Series:
    """Return a boolean mask for null, empty, or whitespace-only entries."""

    return series.isna() | series.astype(str).str.strip().eq("")


def validate_labels(labels: Iterable[object]) -> np.ndarray:
    """Validate and canonicalize labels without accepting missing/unknown classes."""

    normalized: list[str] = []
    for position, label in enumerate(labels):
        if pd.isna(label) or not isinstance(label, str) or not label.strip():
            raise DatasetValidationError(f"Missing gesture label at position {position}.")
        try:
            normalized.append(canonicalize_gesture_label(label))
        except ValueError as exc:
            raise DatasetValidationError(
                f"Unsupported gesture label at position {position}: {label!r}."
            ) from exc
    if not normalized:
        raise DatasetValidationError("Dataset has no labels or no rows.")
    return np.asarray(normalized, dtype=object)


def validate_dataset(
    dataframe: pd.DataFrame,
    *,
    require_all_gestures: bool = False,
    require_unique_sample_ids: bool = True,
) -> pd.DataFrame:
    """Validate and return a canonical, numeric copy of a raw landmark dataset.

    Required columns are the five metadata fields plus all 63 raw coordinates.
    Extra columns are retained to allow non-breaking dataset extensions.  Only
    canonical gesture labels are returned, so downstream class encoders remain
    stable across collection sessions.
    """

    source = _require_dataframe(dataframe)
    if source.empty:
        raise DatasetValidationError("Dataset is empty; collect at least one valid hand sample.")

    missing_columns = [column for column in DATASET_COLUMNS if column not in source.columns]
    if missing_columns:
        preview = ", ".join(missing_columns[:8])
        suffix = "..." if len(missing_columns) > 8 else ""
        raise DatasetValidationError(
            f"Dataset is missing {len(missing_columns)} required column(s): {preview}{suffix}"
        )

    validated = source.copy()
    for column in ("sample_id", "session_id", "subject_id"):
        invalid = _missing_or_blank(validated[column])
        if invalid.any():
            bad_rows = validated.index[invalid].tolist()[:5]
            raise DatasetValidationError(
                f"Column {column!r} contains missing/blank values at rows {bad_rows}."
            )
        # Metadata are IDs, not numeric quantities.  Strings prevent values
        # such as session 01 losing their leading zero during CSV parsing.
        validated[column] = validated[column].astype(str).str.strip()

    if require_unique_sample_ids and validated["sample_id"].duplicated().any():
        duplicate_ids = validated.loc[
            validated["sample_id"].duplicated(keep=False), "sample_id"
        ].head(5).tolist()
        raise DatasetValidationError(
            f"sample_id values must be unique; duplicate example(s): {duplicate_ids}."
        )

    try:
        validated["gesture"] = validate_labels(validated["gesture"])
    except DatasetValidationError:
        raise

    if require_all_gestures:
        observed = set(validated["gesture"])
        absent = [label for label in GESTURE_LABELS if label not in observed]
        if absent:
            raise DatasetValidationError(
                "Dataset is missing required gesture class(es): " + ", ".join(absent)
            )

    timestamp = pd.to_numeric(validated["timestamp"], errors="coerce")
    invalid_timestamps = timestamp.isna() | ~np.isfinite(timestamp)
    if invalid_timestamps.any():
        bad_rows = validated.index[invalid_timestamps].tolist()[:5]
        raise DatasetValidationError(
            f"timestamp must be finite numeric values; invalid rows: {bad_rows}."
        )
    validated["timestamp"] = timestamp.astype(float)

    numeric_features = validated.loc[:, list(RAW_FEATURE_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_features = numeric_features.isna() | ~np.isfinite(numeric_features)
    if invalid_features.to_numpy().any():
        locations = np.argwhere(invalid_features.to_numpy())[:5]
        examples = [
            f"row {validated.index[row_index]!r}, column {RAW_FEATURE_COLUMNS[column_index]!r}"
            for row_index, column_index in locations
        ]
        raise DatasetValidationError(
            "Raw landmark coordinates must be finite numeric values; invalid "
            f"example(s): {', '.join(examples)}."
        )
    validated.loc[:, list(RAW_FEATURE_COLUMNS)] = numeric_features.astype(float)
    return validated


def load_dataset(
    path: str | Path,
    *,
    require_all_gestures: bool = False,
    require_unique_sample_ids: bool = True,
) -> pd.DataFrame:
    """Read and validate a GestureX raw-landmark CSV from ``path``."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Run src/collect_data.py first or pass --data."
        )
    if not csv_path.is_file():
        raise DatasetValidationError(f"Dataset path is not a file: {csv_path}.")
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise DatasetValidationError(f"Could not read CSV dataset {csv_path}: {exc}") from exc
    return validate_dataset(
        frame,
        require_all_gestures=require_all_gestures,
        require_unique_sample_ids=require_unique_sample_ids,
    )


load_landmark_dataset = load_dataset


def class_distribution(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Report observed class counts, percentages, and the global imbalance ratio.

    The result only lists classes that are actually present.  A separate
    ``missing_supported_classes`` attribute makes an incomplete collection
    visible without pretending that uncollected classes have real zero samples.
    """

    validated = validate_dataset(dataframe)
    counts = validated["gesture"].value_counts()
    ordered_labels = [label for label in GESTURE_LABELS if label in counts.index]
    counts = counts.reindex(ordered_labels)
    total = int(counts.sum())
    smallest = int(counts.min())
    imbalance_ratio = float(counts.max() / smallest) if smallest else float("inf")
    report = pd.DataFrame(
        {
            "gesture": counts.index,
            "count": counts.to_numpy(dtype=int),
            "percentage": (counts.to_numpy(dtype=float) / total) * 100.0,
            "imbalance_ratio": np.repeat(imbalance_ratio, len(counts)),
        }
    )
    report.attrs["total_samples"] = total
    report.attrs["imbalance_ratio"] = imbalance_ratio
    report.attrs["missing_supported_classes"] = [
        label for label in GESTURE_LABELS if label not in counts.index
    ]
    return report


get_class_distribution = class_distribution


def class_distribution_summary(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return a serializable class-balance summary for logs or JSON reports."""

    report = class_distribution(dataframe)
    return {
        "total_samples": int(report.attrs["total_samples"]),
        "imbalance_ratio": float(report.attrs["imbalance_ratio"]),
        "missing_supported_classes": list(report.attrs["missing_supported_classes"]),
        "classes": report.to_dict(orient="records"),
    }


def prepare_feature_matrix(
    dataframe: pd.DataFrame,
    representation: str = RAW_REPRESENTATION,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return ``(X, y, metadata)`` without fitting learned preprocessing.

    ``metadata`` retains sample/session/subject IDs for group-aware splitting.
    Never discard it before defining the final cross-session holdout.
    """

    validated = validate_dataset(dataframe)
    selected = normalize_representation(representation)
    try:
        feature_frame = transform_feature_frame(validated, selected)
    except FeatureExtractionError as exc:
        raise DatasetValidationError(str(exc)) from exc
    expected_columns = feature_names(selected)
    if tuple(feature_frame.columns) != tuple(expected_columns):
        raise DatasetValidationError(
            "Feature extraction returned an unexpected column ordering or dimensionality."
        )
    features = feature_frame.to_numpy(dtype=np.float64, copy=True)
    if features.ndim != 2 or features.shape[1] != len(expected_columns):
        raise DatasetValidationError(
            f"Expected a feature matrix with {len(expected_columns)} columns, got {features.shape}."
        )
    labels = validated["gesture"].to_numpy(dtype=object, copy=True)
    metadata = validated.loc[:, list(METADATA_COLUMNS)].copy()
    return features, labels, metadata


get_feature_matrix = prepare_feature_matrix


def build_feature_dataset(
    dataframe: pd.DataFrame,
    representation: str = RAW_REPRESENTATION,
) -> pd.DataFrame:
    """Return metadata plus one selected representation in a CSV-ready DataFrame."""

    validated = validate_dataset(dataframe)
    selected = normalize_representation(representation)
    feature_frame = transform_feature_frame(validated, selected)
    return pd.concat(
        [validated.loc[:, list(METADATA_COLUMNS)].copy(), feature_frame], axis=1
    )


def _normalise_session_ids(session_ids: str | Iterable[object]) -> set[str]:
    if isinstance(session_ids, str):
        requested = {session_ids.strip()}
    else:
        requested = {str(value).strip() for value in session_ids}
    requested.discard("")
    if not requested:
        raise DatasetValidationError("At least one non-empty held-out session ID is required.")
    return requested


def split_by_session(
    dataframe: pd.DataFrame,
    test_session_ids: str | Iterable[object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create explicit train/test partitions with non-overlapping sessions.

    This is deliberately *not* a random row split.  Adjacent webcam frames are
    highly correlated; mixing them across partitions would overstate accuracy.
    """

    validated = validate_dataset(dataframe)
    requested = _normalise_session_ids(test_session_ids)
    available = set(validated["session_id"])
    unavailable = sorted(requested - available)
    if unavailable:
        raise DatasetValidationError(
            "Requested held-out session(s) are absent from the dataset: "
            + ", ".join(unavailable)
        )

    test_mask = validated["session_id"].isin(requested)
    train_frame = validated.loc[~test_mask].copy()
    test_frame = validated.loc[test_mask].copy()
    if train_frame.empty:
        raise DatasetValidationError("Session split leaves no training samples.")
    if test_frame.empty:  # Defensive after the availability check.
        raise DatasetValidationError("Session split leaves no held-out test samples.")
    assert_session_separation(train_frame, test_frame)
    return train_frame, test_frame


session_aware_split = split_by_session


def assert_session_separation(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:
    """Raise if training and test samples share any recording session ID."""

    train = _require_dataframe(train_dataframe)
    test = _require_dataframe(test_dataframe)
    for name, frame in (("train", train), ("test", test)):
        if "session_id" not in frame.columns:
            raise DatasetValidationError(f"{name} split has no session_id column.")
        if _missing_or_blank(frame["session_id"]).any():
            raise DatasetValidationError(f"{name} split contains a missing session ID.")
    overlap = set(train["session_id"].astype(str)) & set(test["session_id"].astype(str))
    if overlap:
        raise DatasetValidationError(
            "Session leakage detected: train and test share session ID(s): "
            + ", ".join(sorted(overlap))
        )


validate_session_separation = assert_session_separation


if __name__ == "__main__":  # pragma: no cover - small usability helper.
    import argparse

    parser = argparse.ArgumentParser(description="Validate a GestureX landmark CSV.")
    parser.add_argument("data", type=Path, help="Path to a raw GestureX CSV file.")
    arguments = parser.parse_args()
    loaded = load_dataset(arguments.data)
    print(class_distribution(loaded).to_string(index=False))
