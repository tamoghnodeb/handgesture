"""Leakage-aware model selection for GestureX.

The cross-session partition is intentionally never read by the model-selection
routine.  Candidate models are selected with grouped, time-ordered validation
folds drawn only from the development sessions.  This matters for webcam data:
adjacent frames can be nearly identical, so a random row split can report an
unrealistically high score.

Run from the repository root, for example::

    python src/train.py --data data/processed/landmarks.csv \
        --cross-session session_02

After training, run :mod:`src.evaluate` to produce the held-out same-session
and cross-session results.  ``train.py`` deliberately does *not* score the
cross-session rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIONS = ("raw", "invariant")
MODEL_NAMES = ("logistic_regression", "random_forest", "rbf_svm")
REQUIRED_METADATA = ("sample_id", "session_id", "subject_id", "gesture", "timestamp")
DEFAULT_GESTURE_LABELS = (
    "open_palm",
    "fist",
    "thumbs_up",
    "thumbs_down",
    "victory",
    "point_left",
    "point_right",
)


class ExperimentError(ValueError):
    """An actionable error caused by an invalid experiment configuration."""


@dataclass(frozen=True)
class SessionSplit:
    """Development and held-out partitions produced by the blocked splitter."""

    train: pd.DataFrame
    same_session_test: pd.DataFrame
    temporal_guard: pd.DataFrame
    cross_session_test: pd.DataFrame
    temporal_groups: pd.Series


def _import_module(name: str) -> Any | None:
    """Import a sibling module in both ``python -m`` and script execution modes."""

    try:
        if __package__:
            return __import__(f"{__package__}.{name}", fromlist=[name])
        return __import__(name)
    except ImportError:
        return None


def _config_value(*names: str, default: Any = None) -> Any:
    config = _import_module("config")
    if config is None:
        return default
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def get_gesture_labels() -> tuple[str, ...]:
    """Return the configured labels, with a safe standalone fallback."""

    labels = _config_value("GESTURE_LABELS", "GESTURE_CLASSES", default=DEFAULT_GESTURE_LABELS)
    if isinstance(labels, dict):
        labels = tuple(labels.keys())
    normalized = tuple(str(label) for label in labels)
    if not normalized:
        raise ExperimentError("The configured gesture-label list is empty.")
    return normalized


def raw_feature_columns() -> list[str]:
    """Return the documented flattened ``x0,y0,z0,...,x20,y20,z20`` ordering."""

    configured = _config_value("RAW_FEATURE_COLUMNS", default=None)
    if configured is not None:
        columns = [str(column) for column in configured]
    else:
        columns = [f"{axis}{index}" for index in range(21) for axis in ("x", "y", "z")]
    if len(columns) != 63:
        raise ExperimentError(
            f"Raw landmark schema must have 63 feature columns, found {len(columns)}."
        )
    return columns


def invariant_feature_columns() -> list[str]:
    """Get names for the eight geometry features without assuming config internals."""

    configured = _config_value("INVARIANT_FEATURE_COLUMNS", default=None)
    if configured is not None:
        columns = [str(column) for column in configured]
    else:
        columns = [
            "thumb_tip_to_index_mcp_norm",
            "index_tip_to_index_mcp_norm",
            "middle_tip_to_middle_mcp_norm",
            "ring_tip_to_ring_mcp_norm",
            "pinky_tip_to_pinky_mcp_norm",
            "thumb_angle",
            "index_angle",
            "middle_angle",
        ]
    if len(columns) != 8:
        raise ExperimentError(
            f"Invariant feature schema must have 8 columns, found {len(columns)}."
        )
    return columns


def default_data_path() -> Path:
    """Choose a configured dataset path, if one exists, without inventing data."""

    configured = _config_value(
        "DATASET_PATH",
        "PROCESSED_DATA_PATH",
        "PROCESSED_DATASET_PATH",
        "RAW_DATASET_PATH",
        "DEFAULT_DATASET_PATH",
        default=None,
    )
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate
    return PROJECT_ROOT / "data" / "processed" / "landmarks.csv"


def default_models_dir() -> Path:
    value = _config_value("MODELS_DIR", default=PROJECT_ROOT / "models")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_results_dir() -> Path:
    value = _config_value("RESULTS_DIR", default=PROJECT_ROOT / "results")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configured_best_model_path(models_dir: Path) -> Path:
    """Return the config's conventional deploy artifact path when it is configured."""

    value = _config_value("MODEL_ARTIFACT_PATH", default=models_dir / "gesturex_model.joblib")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_dataset(data_path: str | Path) -> pd.DataFrame:
    """Read and strictly validate a landmark CSV before any split is made."""

    path = Path(data_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raw_dir = path if path.is_dir() else (PROJECT_ROOT / "data" / "raw")
        raw_csvs = sorted(raw_dir.glob("*.csv")) if raw_dir.is_dir() else []
        session_csvs = [p for p in raw_csvs if p.name.startswith("session_")] or raw_csvs
        if session_csvs:
            try:
                frames = [pd.read_csv(p) for p in session_csvs]
                dataframe = pd.concat(frames, ignore_index=True)
                combined_raw = PROJECT_ROOT / "data" / "raw" / "gesture_landmarks.csv"
                combined_proc = PROJECT_ROOT / "data" / "processed" / "gesture_features.csv"
                combined_raw.parent.mkdir(parents=True, exist_ok=True)
                combined_proc.parent.mkdir(parents=True, exist_ok=True)
                dataframe.to_csv(combined_raw, index=False)
                dataframe.to_csv(combined_proc, index=False)
            except Exception as exc:
                raise ExperimentError(f"Could not aggregate raw CSV files in '{raw_dir}': {exc}") from exc
        else:
            raise ExperimentError(
                f"Dataset not found: {path}. Collect data first or pass --data PATH."
            )
    else:
        try:
            dataframe = pd.read_csv(path)
        except Exception as exc:  # pandas provides the useful lower-level detail.
            raise ExperimentError(f"Could not read CSV '{path}': {exc}") from exc

    missing = [column for column in REQUIRED_METADATA if column not in dataframe.columns]
    if missing:
        raise ExperimentError(
            "Dataset is missing required metadata columns: "
            f"{', '.join(missing)}. Recollect data with src/collect_data.py."
        )
    if dataframe.empty:
        raise ExperimentError("Dataset has no samples; there are no results to train.")
    if dataframe.loc[:, REQUIRED_METADATA].isna().any().any():
        null_columns = dataframe.loc[:, REQUIRED_METADATA].columns[
            dataframe.loc[:, REQUIRED_METADATA].isna().any()
        ].tolist()
        raise ExperimentError(
            f"Dataset has missing required metadata values in: {', '.join(null_columns)}."
        )

    dataframe = dataframe.copy()
    for column in REQUIRED_METADATA:
        dataframe[column] = dataframe[column].astype(str)
    if dataframe["sample_id"].duplicated().any():
        examples = dataframe.loc[dataframe["sample_id"].duplicated(), "sample_id"].head(3).tolist()
        raise ExperimentError(
            "sample_id values must be unique so split membership can be audited. "
            f"Duplicates include: {examples}."
        )

    configured_labels = set(get_gesture_labels())
    observed_labels = set(dataframe["gesture"])
    unsupported = sorted(observed_labels - configured_labels)
    if unsupported:
        raise ExperimentError(
            "Dataset contains unsupported gesture labels: "
            f"{unsupported}. Configure them explicitly or correct the CSV."
        )

    missing_raw = [column for column in raw_feature_columns() if column not in dataframe.columns]
    if missing_raw:
        raise ExperimentError(
            "Dataset is missing raw landmark coordinate columns, for example: "
            f"{', '.join(missing_raw[:6])}."
        )
    raw_values = dataframe.loc[:, raw_feature_columns()].apply(pd.to_numeric, errors="coerce")
    if raw_values.isna().any().any() or not np.isfinite(raw_values.to_numpy(dtype=float)).all():
        raise ExperimentError(
            "Raw landmark coordinates must all be finite numeric values; invalid samples "
            "should be removed or recollected rather than silently imputed."
        )
    dataframe.loc[:, raw_feature_columns()] = raw_values
    return dataframe


def parse_session_list(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse comma-separated CLI session IDs while retaining user-provided IDs exactly."""

    if value is None:
        return ()
    values = [value] if isinstance(value, str) else value
    parsed: list[str] = []
    for item in values:
        parsed.extend(part.strip() for part in str(item).split(",") if part.strip())
    return tuple(dict.fromkeys(parsed))


def _stable_time_order(group: pd.DataFrame) -> pd.DataFrame:
    """Order a recording deterministically by timestamp, falling back to CSV order."""

    ordered = group.copy()
    numeric_timestamp = pd.to_numeric(ordered["timestamp"], errors="coerce")
    if numeric_timestamp.notna().all():
        return ordered.assign(_numeric_timestamp=numeric_timestamp).sort_values(
            ["_numeric_timestamp", "sample_id"], kind="stable"
        ).drop(columns="_numeric_timestamp")

    parsed_timestamp = pd.to_datetime(ordered["timestamp"], errors="coerce", utc=True)
    if parsed_timestamp.notna().all():
        ordered = ordered.assign(_parsed_timestamp=parsed_timestamp).sort_values(
            ["_parsed_timestamp", "sample_id"], kind="stable"
        )
        return ordered.drop(columns="_parsed_timestamp")
    # Collection order is still far safer than randomly mixing adjacent camera frames.
    return ordered.sort_index(kind="stable")


def build_blocked_session_split(
    dataframe: pd.DataFrame,
    *,
    cross_sessions: Sequence[str],
    development_sessions: Sequence[str] | None = None,
    holdout_fraction: float = 0.20,
    temporal_blocks: int = 8,
    guard_blocks: int = 1,
    min_samples_per_block: int = 5,
) -> SessionSplit:
    """Create a session-aware split with a temporal gap around the same-session test.

    The final contiguous block(s) of *each (session, gesture)* recording become
    the same-session test set.  The preceding block(s) are discarded as a guard
    band.  Therefore frames on either side of a split never enter train and test
    together.  The distinct ``cross_sessions`` are excluded from both partition
    creation and model selection.
    """

    if not 0 < holdout_fraction < 1:
        raise ExperimentError("--same-session-holdout must be between 0 and 1.")
    if temporal_blocks < 3:
        raise ExperimentError("--temporal-blocks must be at least 3.")
    if guard_blocks < 0:
        raise ExperimentError("--guard-blocks cannot be negative.")
    if min_samples_per_block < 1:
        raise ExperimentError("--min-samples-per-block must be at least 1.")

    available_sessions = set(dataframe["session_id"])
    cross_sessions = tuple(cross_sessions)
    if not cross_sessions:
        raise ExperimentError(
            "An explicit --cross-session SESSION_ID is required. This prevents an "
            "accidental random split from being presented as cross-session performance."
        )
    unknown_cross = sorted(set(cross_sessions) - available_sessions)
    if unknown_cross:
        raise ExperimentError(f"Unknown cross-session ID(s): {unknown_cross}.")

    if development_sessions:
        development_sessions = tuple(development_sessions)
        unknown_development = sorted(set(development_sessions) - available_sessions)
        if unknown_development:
            raise ExperimentError(f"Unknown development session ID(s): {unknown_development}.")
        overlap = set(development_sessions) & set(cross_sessions)
        if overlap:
            raise ExperimentError(
                "A session cannot be in both development and cross-session sets: "
                f"{sorted(overlap)}."
            )
    else:
        development_sessions = tuple(sorted(available_sessions - set(cross_sessions)))

    if not development_sessions:
        raise ExperimentError("No development sessions remain after holding out cross-session data.")

    development = dataframe.loc[dataframe["session_id"].isin(development_sessions)].copy()
    cross = dataframe.loc[dataframe["session_id"].isin(cross_sessions)].copy()
    if cross.empty:
        raise ExperimentError("Cross-session partition is empty.")

    partitions = pd.Series("unassigned", index=development.index, dtype="object")
    temporal_group = pd.Series("", index=development.index, dtype="object")
    group_columns = ["session_id", "gesture"]
    for (session_id, gesture), group in development.groupby(group_columns, sort=True):
        ordered = _stable_time_order(group)
        sample_count = len(ordered)
        block_count = min(temporal_blocks, sample_count // min_samples_per_block)
        # train + guard + test must each contain at least one block.
        if block_count < guard_blocks + 2:
            raise ExperimentError(
                "Not enough time-ordered samples for leakage-safe same-session splitting "
                f"in session '{session_id}', gesture '{gesture}' ({sample_count} samples). "
                "Collect more samples, reduce --guard-blocks, or reduce "
                "--min-samples-per-block."
            )
        test_blocks = max(1, int(np.ceil(block_count * holdout_fraction)))
        while block_count - test_blocks - guard_blocks < 1:
            test_blocks -= 1
        if test_blocks < 1:
            raise ExperimentError(
                f"Could not reserve train/guard/test blocks for {session_id}/{gesture}."
            )

        positions = np.arange(sample_count)
        block_ids = np.floor(positions * block_count / sample_count).astype(int)
        train_limit = block_count - test_blocks - guard_blocks
        group_prefix = f"{session_id}::{gesture}"
        temporal_group.loc[ordered.index] = [f"{group_prefix}::block_{block}" for block in block_ids]
        partitions.loc[ordered.index[block_ids < train_limit]] = "same_session_train"
        if guard_blocks:
            partitions.loc[
                ordered.index[(block_ids >= train_limit) & (block_ids < block_count - test_blocks)]
            ] = "temporal_guard"
        partitions.loc[ordered.index[block_ids >= block_count - test_blocks]] = "same_session_test"

    if (partitions == "unassigned").any():
        raise ExperimentError("Internal split error: some development samples were not assigned.")

    train = development.loc[partitions.eq("same_session_train")].copy()
    same_test = development.loc[partitions.eq("same_session_test")].copy()
    guard = development.loc[partitions.eq("temporal_guard")].copy()
    if train.empty or same_test.empty:
        raise ExperimentError("Leakage-safe split produced an empty train or same-session test set.")

    train_labels = set(train["gesture"])
    test_labels = set(same_test["gesture"])
    if train_labels != test_labels:
        raise ExperimentError(
            "The blocked same-session split does not retain every gesture in both train "
            "and test. Collect more balanced samples per gesture."
        )

    return SessionSplit(
        train=train,
        same_session_test=same_test,
        temporal_guard=guard,
        cross_session_test=cross,
        temporal_groups=temporal_group.loc[train.index],
    )


def feature_matrix(dataframe: pd.DataFrame, representation: str) -> tuple[np.ndarray, list[str]]:
    """Build one documented representation and fail clearly on invalid output."""

    if representation not in REPRESENTATIONS:
        raise ExperimentError(f"Unknown feature representation '{representation}'.")
    if representation == "raw":
        columns = raw_feature_columns()
        matrix = dataframe.loc[:, columns].to_numpy(dtype=np.float64, copy=True)
    else:
        features_module = _import_module("features")
        if features_module is None or not hasattr(features_module, "transform_feature_frame"):
            raise ExperimentError(
                "Invariant feature extraction is unavailable. Ensure src/features.py is present."
            )
        try:
            transformed = features_module.transform_feature_frame(dataframe, "invariant")
        except Exception as exc:
            raise ExperimentError(f"Invariant feature extraction failed: {exc}") from exc
        if isinstance(transformed, pd.DataFrame):
            columns = list(transformed.columns)
            matrix = transformed.to_numpy(dtype=np.float64, copy=True)
        else:
            matrix = np.asarray(transformed, dtype=np.float64)
            columns = invariant_feature_columns()

    if matrix.ndim != 2 or matrix.shape[0] != len(dataframe):
        raise ExperimentError(
            f"{representation} feature extractor returned invalid shape {matrix.shape}; "
            f"expected ({len(dataframe)}, n_features)."
        )
    expected_width = 63 if representation == "raw" else 8
    if matrix.shape[1] != expected_width:
        raise ExperimentError(
            f"{representation} feature matrix has {matrix.shape[1]} columns; "
            f"expected {expected_width}."
        )
    if not np.isfinite(matrix).all():
        raise ExperimentError(
            f"{representation} feature matrix contains NaN or infinity; inspect invalid landmarks."
        )
    return matrix, columns


def build_pipeline(model_name: str, random_state: int) -> Pipeline:
    """Create a fresh pipeline; scalers are fitted inside each training fold only."""

    if model_name == "logistic_regression":
        parameters = dict(
            _config_value(
                "LOGISTIC_REGRESSION_PARAMS",
                default={"max_iter": 2_000},
            )
        )
        parameters.pop("multi_class", None)
        parameters["random_state"] = random_state
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(**parameters)),
            ]
        )
    if model_name == "random_forest":
        parameters = dict(
            _config_value(
                "RANDOM_FOREST_PARAMS",
                default={"n_estimators": 300, "n_jobs": -1},
            )
        )
        parameters["random_state"] = random_state
        return Pipeline(
            [
                ("model", RandomForestClassifier(**parameters))
            ]
        )
    if model_name == "rbf_svm":
        parameters = dict(
            _config_value(
                "RBF_SVM_PARAMS",
                default={"kernel": "rbf", "C": 3.0, "gamma": "scale", "probability": True},
            )
        )
        parameters["random_state"] = random_state
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(**parameters)),
            ]
        )
    raise ExperimentError(f"Unknown model '{model_name}'. Expected one of {MODEL_NAMES}.")


def grouped_cv_splits(
    labels: Sequence[str], groups: Sequence[str], requested_folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return stratified *grouped* folds, never a row-wise random split."""

    label_series = pd.Series(labels, dtype="string")
    group_series = pd.Series(groups, dtype="string")
    group_counts = pd.DataFrame({"label": label_series, "group": group_series}).groupby("label")[
        "group"
    ].nunique()
    max_folds = int(group_counts.min()) if not group_counts.empty else 0
    folds = min(int(requested_folds), max_folds)
    if folds < 2:
        details = group_counts.to_dict()
        raise ExperimentError(
            "Grouped validation needs at least two independent temporal blocks for every "
            f"gesture after the same-session holdout; group counts: {details}. Collect "
            "more samples or use fewer/larger temporal blocks."
        )
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    indices = np.arange(len(label_series))
    return list(splitter.split(indices, label_series, group_series))


def _cv_metric_record(
    representation: str,
    model_name: str,
    fold_index: int,
    truth: Sequence[str],
    prediction: Sequence[str],
) -> dict[str, Any]:
    return {
        "feature_representation": representation,
        "model": model_name,
        "fold": fold_index,
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_precision": float(precision_score(truth, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, prediction, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "n_validation_samples": int(len(truth)),
    }


def select_representation_model(
    train_frame: pd.DataFrame,
    temporal_groups: pd.Series,
    representation: str,
    *,
    cv_folds: int,
    random_state: int,
) -> tuple[Pipeline, list[dict[str, Any]], list[str], str]:
    """Select a model with grouped development-only validation and fit it once."""

    matrix, columns = feature_matrix(train_frame, representation)
    labels = train_frame["gesture"].to_numpy(dtype=str)
    groups = temporal_groups.loc[train_frame.index].to_numpy(dtype=str)
    splits = grouped_cv_splits(labels, groups, cv_folds)
    fold_records: list[dict[str, Any]] = []

    for model_name in MODEL_NAMES:
        template = build_pipeline(model_name, random_state)
        for fold_index, (fit_indices, validation_indices) in enumerate(splits, start=1):
            # clone gives every fold a fresh scaler: validation rows cannot affect it.
            pipeline = clone(template)
            pipeline.fit(matrix[fit_indices], labels[fit_indices])
            prediction = pipeline.predict(matrix[validation_indices])
            fold_records.append(
                _cv_metric_record(
                    representation,
                    model_name,
                    fold_index,
                    labels[validation_indices],
                    prediction,
                )
            )

    records_frame = pd.DataFrame(fold_records)
    summary = (
        records_frame.groupby(["feature_representation", "model"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_macro_precision=("macro_precision", "mean"),
            mean_macro_recall=("macro_recall", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            cv_folds=("fold", "count"),
        )
        .sort_values(["mean_macro_f1", "mean_accuracy", "model"], ascending=[False, False, True])
    )
    winner_name = str(summary.iloc[0]["model"])
    fitted_pipeline = build_pipeline(winner_name, random_state)
    # This artifact remains fitted only on the same-session training partition.
    fitted_pipeline.fit(matrix, labels)
    return fitted_pipeline, fold_records, columns, winner_name


def _dataset_fingerprint(dataframe: pd.DataFrame) -> str:
    """Create a stable audit fingerprint for split-manifest compatibility checks."""

    columns = [*REQUIRED_METADATA, *raw_feature_columns()]
    payload = dataframe.loc[:, columns].to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_split_manifest(
    dataframe: pd.DataFrame,
    split: SessionSplit,
    *,
    data_path: Path,
    cross_sessions: Sequence[str],
    development_sessions: Sequence[str],
    holdout_fraction: float,
    temporal_blocks: int,
    guard_blocks: int,
    min_samples_per_block: int,
) -> dict[str, Any]:
    """Store exact row membership so evaluation cannot accidentally reconstruct a split."""

    def ids(frame: pd.DataFrame) -> list[str]:
        return frame["sample_id"].astype(str).tolist()

    return {
        "manifest_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(data_path.resolve()),
        "dataset_fingerprint": _dataset_fingerprint(dataframe),
        "split_strategy": {
            "name": "per_session_per_gesture_time_block_holdout_with_guard_band",
            "same_session_holdout_fraction": holdout_fraction,
            "temporal_blocks_requested": temporal_blocks,
            "guard_blocks": guard_blocks,
            "min_samples_per_block": min_samples_per_block,
            "rationale": (
                "Rows are ordered by timestamp within each session and gesture. Final "
                "contiguous block(s) are same-session test data; preceding blocks are "
                "discarded to prevent adjacent webcam frames from leaking across the split."
            ),
        },
        "development_sessions": list(development_sessions),
        "cross_session_test_sessions": list(cross_sessions),
        "partitions": {
            "same_session_train": ids(split.train),
            "same_session_test": ids(split.same_session_test),
            "temporal_guard_excluded": ids(split.temporal_guard),
            "cross_session_test": ids(split.cross_session_test),
        },
        "counts": {
            "same_session_train": int(len(split.train)),
            "same_session_test": int(len(split.same_session_test)),
            "temporal_guard_excluded": int(len(split.temporal_guard)),
            "cross_session_test": int(len(split.cross_session_test)),
        },
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def train_experiment(
    *,
    data_path: str | Path,
    cross_sessions: Sequence[str],
    development_sessions: Sequence[str] | None = None,
    representations: Iterable[str] = REPRESENTATIONS,
    same_session_holdout: float = 0.20,
    temporal_blocks: int = 8,
    guard_blocks: int = 1,
    min_samples_per_block: int = 5,
    cv_folds: int = 5,
    random_state: int = 42,
    models_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train all required candidate families without touching cross-session rows."""

    normalized_representations = tuple(dict.fromkeys(str(item) for item in representations))
    invalid = sorted(set(normalized_representations) - set(REPRESENTATIONS))
    if invalid or not normalized_representations:
        raise ExperimentError(f"--representations must use {REPRESENTATIONS}; got {invalid}.")

    dataset_path = Path(data_path)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    dataframe = read_dataset(dataset_path)
    normalized_cross_sessions = parse_session_list(cross_sessions)
    normalized_development_sessions = parse_session_list(development_sessions)
    split = build_blocked_session_split(
        dataframe,
        cross_sessions=normalized_cross_sessions,
        development_sessions=normalized_development_sessions or None,
        holdout_fraction=same_session_holdout,
        temporal_blocks=temporal_blocks,
        guard_blocks=guard_blocks,
        min_samples_per_block=min_samples_per_block,
    )
    if not normalized_development_sessions:
        normalized_development_sessions = tuple(sorted(set(dataframe["session_id"]) - set(normalized_cross_sessions)))

    output_models = Path(models_dir) if models_dir else default_models_dir()
    output_results = Path(results_dir) if results_dir else default_results_dir()
    if not output_models.is_absolute():
        output_models = PROJECT_ROOT / output_models
    if not output_results.is_absolute():
        output_results = PROJECT_ROOT / output_results
    output_models.mkdir(parents=True, exist_ok=True)
    output_results.mkdir(parents=True, exist_ok=True)

    manifest = make_split_manifest(
        dataframe,
        split,
        data_path=dataset_path,
        cross_sessions=normalized_cross_sessions,
        development_sessions=normalized_development_sessions,
        holdout_fraction=same_session_holdout,
        temporal_blocks=temporal_blocks,
        guard_blocks=guard_blocks,
        min_samples_per_block=min_samples_per_block,
    )
    manifest_path = output_models / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")

    all_fold_records: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    labels = list(get_gesture_labels())
    for representation in normalized_representations:
        pipeline, fold_records, columns, winner_name = select_representation_model(
            split.train,
            split.temporal_groups,
            representation,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        all_fold_records.extend(fold_records)
        representation_records = [row for row in fold_records if row["model"] == winner_name]
        validation_macro_f1 = float(np.mean([row["macro_f1"] for row in representation_records]))
        validation_accuracy = float(np.mean([row["accuracy"] for row in representation_records]))
        artifact: dict[str, Any] = {
            "artifact_version": 1,
            "project": "GestureX",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline": pipeline,
            "feature_representation": representation,
            "feature_columns": columns,
            "input_feature_dim": len(columns),
            "class_labels": labels,
            "best_model_name": winner_name,
            "selection_metric": "mean grouped validation macro F1",
            "validation_macro_f1": validation_macro_f1,
            "validation_accuracy": validation_accuracy,
            "training_scope": (
                "Fitted only on same_session_train. The same-session test, temporal guard "
                "band, and cross-session test were excluded from this saved pipeline."
            ),
            "split_manifest": str(manifest_path.resolve()),
            "training_sessions": list(normalized_development_sessions),
            "cross_session_test_sessions": list(normalized_cross_sessions),
            "random_state": random_state,
        }
        artifact_path = output_models / f"gesturex_{representation}_best.joblib"
        joblib.dump(artifact, artifact_path)
        artifacts[representation] = {
            "artifact": artifact,
            "path": artifact_path,
            "validation_macro_f1": validation_macro_f1,
            "validation_accuracy": validation_accuracy,
        }

    fold_frame = pd.DataFrame(all_fold_records)
    fold_path = output_results / "training_validation_folds.csv"
    fold_frame.to_csv(fold_path, index=False)
    validation_summary = (
        fold_frame.groupby(["feature_representation", "model"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_macro_precision=("macro_precision", "mean"),
            mean_macro_recall=("macro_recall", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            cv_folds=("fold", "count"),
        )
        .sort_values(
            ["mean_macro_f1", "mean_accuracy", "feature_representation", "model"],
            ascending=[False, False, True, True],
        )
    )
    validation_summary_path = output_results / "training_validation_summary.csv"
    validation_summary.to_csv(validation_summary_path, index=False)

    best_representation = max(
        artifacts,
        key=lambda item: (
            artifacts[item]["validation_macro_f1"],
            artifacts[item]["validation_accuracy"],
        ),
    )
    best_path = output_models / "gesturex_best.joblib"
    # The best artifact is an exact copy of a representation-specific pipeline, not a refit
    # that would accidentally absorb held-out data.
    joblib.dump(artifacts[best_representation]["artifact"], best_path)
    # Keep config.MODEL_ARTIFACT_PATH as a convenience alias for deployment while
    # retaining the descriptive, representation-specific artifacts for auditing.
    configured_path = configured_best_model_path(output_models)
    configured_path.parent.mkdir(parents=True, exist_ok=True)
    if configured_path.resolve() != best_path.resolve():
        joblib.dump(artifacts[best_representation]["artifact"], configured_path)

    class_distribution = (
        split.train["gesture"].value_counts().rename_axis("gesture").reset_index(name="train_samples")
    )
    class_distribution["train_percentage"] = (
        100 * class_distribution["train_samples"] / class_distribution["train_samples"].sum()
    )
    class_distribution_path = output_results / "training_class_distribution.csv"
    class_distribution.to_csv(class_distribution_path, index=False)

    training_summary = {
        "best_representation": best_representation,
        "best_model": artifacts[best_representation]["artifact"]["best_model_name"],
        "best_validation_macro_f1": artifacts[best_representation]["validation_macro_f1"],
        "best_validation_accuracy": artifacts[best_representation]["validation_accuracy"],
        "best_artifact": str(best_path.resolve()),
        "deployment_artifact": str(configured_path.resolve()),
        "representation_artifacts": {key: str(value["path"].resolve()) for key, value in artifacts.items()},
        "split_manifest": str(manifest_path.resolve()),
        "cross_session_data_used_for_selection": False,
        "next_step": "Run python src/evaluate.py with the same --data path to measure held-out results.",
    }
    summary_path = output_results / "training_summary.json"
    summary_path.write_text(json.dumps(training_summary, indent=2, default=_json_default), encoding="utf-8")
    return training_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train GestureX models with session-aware, leakage-safe validation."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=default_data_path(),
        help="Combined landmark CSV produced by data collection/preprocessing.",
    )
    parser.add_argument(
        "--cross-session",
        required=True,
        help="Comma-separated session ID(s) reserved exclusively for final evaluation.",
    )
    parser.add_argument(
        "--development-sessions",
        default=None,
        help="Optional comma-separated training/validation session IDs; defaults to all others.",
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=list(REPRESENTATIONS),
        help="Feature representations to compare (default: raw invariant).",
    )
    parser.add_argument("--same-session-holdout", type=float, default=0.20)
    parser.add_argument("--temporal-blocks", type=int, default=8)
    parser.add_argument("--guard-blocks", type=int, default=1)
    parser.add_argument("--min-samples-per-block", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--models-dir", type=Path, default=default_models_dir())
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = train_experiment(
            data_path=args.data,
            cross_sessions=parse_session_list(args.cross_session),
            development_sessions=parse_session_list(args.development_sessions),
            representations=args.representations,
            same_session_holdout=args.same_session_holdout,
            temporal_blocks=args.temporal_blocks,
            guard_blocks=args.guard_blocks,
            min_samples_per_block=args.min_samples_per_block,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            models_dir=args.models_dir,
            results_dir=args.results_dir,
        )
    except ExperimentError as exc:
        parser.error(str(exc))
    print(
        "Training complete without reading cross-session test features. "
        f"Best validation choice: {summary['best_representation']} / {summary['best_model']}.\n"
        f"Saved deployable pipeline: {summary['best_artifact']}\n"
        "Run src/evaluate.py next for actual held-out same-session and cross-session results."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
