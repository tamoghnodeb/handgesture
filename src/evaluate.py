"""Offline same-session and cross-session evaluation for GestureX.

Loads the split manifest and joblib artifacts produced by ``src/train.py``,
evaluates each representation (raw and invariant) on:

1. **Same-session test set** — held-out contiguous blocks from development sessions.
2. **Cross-session test set** — one or more distinct recording sessions.

Generates:

- ``results/raw_same_session.csv``
- ``results/raw_cross_session.csv``
- ``results/invariant_same_session.csv``
- ``results/invariant_cross_session.csv``
- ``results/generalization_results.csv``  (the required four-cell table)
- ``results/confusion_matrix_raw.png``
- ``results/confusion_matrix_invariant.png``
- ``results/class_distribution.png``
- ``results/generalization_comparison.png``
- ``results/evaluation_summary.txt``

Leakage safeguards
------------------
- Split membership is read from the manifest written by ``train.py``; this
  script cannot accidentally reconstruct a different split.
- The scaler embedded in each saved pipeline was fitted only on same-session
  training rows.  These held-out sets are never used to fit any preprocessing.
- Cross-session samples are never inspected during model selection.
- Temporal smoothing is **not** applied; metrics reflect per-frame accuracy.

Usage::

    python src/evaluate.py \\
        --data data/processed/landmarks.csv \\
        --manifest models/split_manifest.json

or simply (if paths match config defaults)::

    python src/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/CI environments.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Repository bootstrap (allows ``python src/evaluate.py`` from root).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    GESTURE_DISPLAY_NAMES,
    GESTURE_LABELS,
    MODELS_DIR,
    RESULTS_DIR,
)
from src.features import transform_feature_frame  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPRESENTATIONS = ("raw", "invariant")
_DISPLAY = {label: GESTURE_DISPLAY_NAMES.get(label, label) for label in GESTURE_LABELS}


class EvaluationError(ValueError):
    """Raised when evaluation cannot proceed due to a configuration problem."""


# ---------------------------------------------------------------------------
# Manifest and dataset utilities
# ---------------------------------------------------------------------------
def load_manifest(path: Path) -> dict[str, Any]:
    """Read and lightly validate the split manifest from ``train.py``."""

    if not path.exists():
        raise EvaluationError(
            f"Split manifest not found: {path}. "
            "Run ``python src/train.py`` before evaluating."
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Manifest is not valid JSON ({path}): {exc}") from exc
    if not isinstance(manifest, dict):
        raise EvaluationError("Manifest must be a JSON object.")
    required = ("partitions", "cross_session_test_sessions", "development_sessions")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise EvaluationError(
            f"Manifest is missing expected keys: {', '.join(missing)}. "
            "Re-run ``python src/train.py`` to regenerate a valid manifest."
        )
    return manifest


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Read the raw landmark CSV (must exist; validation is performed here)."""

    if not data_path.exists():
        raw_dir = PROJECT_ROOT / "data" / "raw"
        raw_csvs = sorted(raw_dir.glob("*.csv")) if raw_dir.is_dir() else []
        session_csvs = [p for p in raw_csvs if p.name.startswith("session_")] or raw_csvs
        if session_csvs:
            try:
                frame = pd.concat([pd.read_csv(p) for p in session_csvs], ignore_index=True)
            except Exception as exc:
                raise EvaluationError(f"Could not aggregate dataset CSVs ({raw_dir}): {exc}") from exc
        else:
            raise EvaluationError(
                f"Dataset not found: {data_path}. "
                "Pass --data PATH or run ``python src/collect_data.py`` first."
            )
    else:
        try:
            frame = pd.read_csv(data_path)
        except Exception as exc:
            raise EvaluationError(f"Could not read dataset CSV ({data_path}): {exc}") from exc
    if "sample_id" not in frame.columns:
        raise EvaluationError("Dataset has no 'sample_id' column; check the CSV schema.")
    frame["sample_id"] = frame["sample_id"].astype(str)
    return frame


def partition_from_manifest(
    dataset: pd.DataFrame, manifest: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, same_session_test, cross_session_test) DataFrames.

    Row membership is read from the manifest; evaluation cannot accidentally
    use a different split than training used.
    """

    partitions: dict[str, list[str]] = manifest["partitions"]
    train_ids = set(partitions.get("same_session_train", []))
    same_ids = set(partitions.get("same_session_test", []))
    cross_ids = set(partitions.get("cross_session_test", []))

    # Verify no overlap — belt-and-suspenders check on top of train.py assertions.
    if train_ids & same_ids:
        raise EvaluationError(
            "Manifest error: same-session train and test share sample IDs (leakage risk)."
        )
    if (train_ids | same_ids) & cross_ids:
        raise EvaluationError(
            "Manifest error: cross-session test overlaps with development data (leakage risk)."
        )

    dataset_ids = set(dataset["sample_id"].astype(str))
    for name, ids in (("same_session_test", same_ids), ("cross_session_test", cross_ids)):
        missing = ids - dataset_ids
        if missing:
            example = sorted(missing)[:5]
            raise EvaluationError(
                f"Dataset is missing {len(missing)} sample IDs listed in the manifest "
                f"partition '{name}', e.g. {example}. "
                "Ensure --data points to the same CSV used during training."
            )

    train_frame = dataset.loc[dataset["sample_id"].isin(train_ids)].copy()
    same_frame = dataset.loc[dataset["sample_id"].isin(same_ids)].copy()
    cross_frame = dataset.loc[dataset["sample_id"].isin(cross_ids)].copy()

    if train_frame.empty:
        raise EvaluationError("Training partition is empty after joining with manifest IDs.")
    if same_frame.empty:
        raise EvaluationError("Same-session test partition is empty after joining with manifest.")
    if cross_frame.empty:
        raise EvaluationError("Cross-session test partition is empty after joining with manifest.")

    return train_frame, same_frame, cross_frame


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def build_feature_matrix(dataset: pd.DataFrame, representation: str) -> np.ndarray:
    """Extract the selected representation from a dataset slice."""

    try:
        feature_frame = transform_feature_frame(dataset, representation)
    except Exception as exc:
        raise EvaluationError(
            f"Feature extraction failed for representation '{representation}': {exc}"
        ) from exc
    return feature_frame.to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
def load_artifact(models_dir: Path, representation: str) -> Any:
    """Load the validation-selected pipeline for one representation."""

    path = models_dir / f"gesturex_{representation}_best.joblib"
    if not path.exists():
        raise EvaluationError(
            f"No trained artifact found for representation '{representation}' at {path}. "
            "Run ``python src/train.py`` first."
        )
    artifact = joblib.load(path)
    if isinstance(artifact, dict):
        pipeline = artifact.get("pipeline") or artifact.get("model")
    else:
        pipeline = artifact
    if pipeline is None:
        raise EvaluationError(
            f"Artifact at {path} does not contain a recognizable sklearn pipeline."
        )
    return pipeline


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _compute_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    labels: Sequence[str],
    split_name: str,
    representation: str,
) -> dict[str, Any]:
    """Return a metrics dictionary for one evaluation split."""

    present_labels = sorted(set(truth) | set(prediction))
    acc = float(accuracy_score(truth, prediction))
    macro_f1 = float(f1_score(truth, prediction, average="macro", zero_division=0, labels=present_labels))
    macro_prec = float(precision_score(truth, prediction, average="macro", zero_division=0, labels=present_labels))
    macro_rec = float(recall_score(truth, prediction, average="macro", zero_division=0, labels=present_labels))
    return {
        "feature_representation": representation,
        "split": split_name,
        "n_samples": int(len(truth)),
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
    }


def _full_report(truth: np.ndarray, prediction: np.ndarray, labels: Sequence[str]) -> str:
    """Generate a classification report using display names."""

    display_labels = [_DISPLAY.get(label, label) for label in labels if label in set(truth) | set(prediction)]
    target_names_ordered = [_DISPLAY.get(label, label) for label in sorted(set(truth) | set(prediction))]
    return classification_report(
        truth,
        prediction,
        labels=sorted(set(truth) | set(prediction)),
        target_names=target_names_ordered,
        zero_division=0,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _save_confusion_matrix(
    truth: np.ndarray,
    prediction: np.ndarray,
    labels: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    """Plot and save a labelled, normalised confusion matrix."""

    present = sorted(set(truth) | set(prediction))
    display_present = [_DISPLAY.get(label, label) for label in present]
    cm = confusion_matrix(truth, prediction, labels=present)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(max(6, len(present)), max(5, len(present))))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        xticklabels=display_present,
        yticklabels=display_present,
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Normalised proportion"},
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(title, fontsize=12, pad=14)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _save_class_distribution(dataset: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of samples per gesture across the entire dataset."""

    counts = dataset["gesture"].value_counts().reindex(GESTURE_LABELS).fillna(0).astype(int)
    display_labels = [_DISPLAY.get(label, label) for label in counts.index]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(display_labels, counts.values, color=sns.color_palette("muted", len(counts)))
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Gesture class", fontsize=11)
    ax.set_ylabel("Sample count", fontsize=11)
    ax.set_title("Class distribution across all sessions", fontsize=12)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _save_generalization_comparison(generalization: pd.DataFrame, output_path: Path) -> None:
    """Grouped bar chart comparing same-session vs cross-session accuracy per representation."""

    representations = generalization["feature_representation"].tolist()
    same_vals = generalization["same_session_accuracy"].tolist()
    cross_vals = generalization["cross_session_accuracy"].tolist()
    x = np.arange(len(representations))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, same_vals, width, label="Same-session", color="#4c72b0")
    bars2 = ax.bar(x + width / 2, cross_vals, width, label="Cross-session", color="#dd8452")
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", " ").title() for r in representations])
    ax.set_xlabel("Feature representation", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Same-session vs cross-session generalization", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Class balance report
# ---------------------------------------------------------------------------
def _class_balance_report(dataset: pd.DataFrame) -> str:
    """Return a text table with counts, percentages, and imbalance ratio."""

    counts = dataset["gesture"].value_counts().reindex(GESTURE_LABELS).fillna(0).astype(int)
    total = int(counts.sum())
    if total == 0:
        return "  (no samples)\n"
    smallest = int(counts.min())
    imbalance = float(counts.max()) / max(smallest, 1)

    lines = ["  Gesture                     Count   Percentage", "  " + "-" * 48]
    for label, count in counts.items():
        display = _DISPLAY.get(label, label)
        pct = 100.0 * count / total
        lines.append(f"  {display:<28} {count:>5}   {pct:>8.2f}%")
    lines.append("  " + "-" * 48)
    lines.append(f"  Total                        {total:>5}   100.00%")
    lines.append(f"  Imbalance ratio (max/min):   {imbalance:.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------
def evaluate(
    *,
    data_path: Path,
    manifest_path: Path,
    models_dir: Path,
    results_dir: Path,
) -> dict[str, Any]:
    """Run the full evaluation and save all required outputs."""

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GestureX offline evaluation")
    print("=" * 70)

    print(f"\nLoading dataset:  {data_path}")
    dataset = load_dataset(data_path)
    print(f"  {len(dataset):,} samples loaded.")

    print(f"\nLoading manifest: {manifest_path}")
    manifest = load_manifest(manifest_path)
    train_frame, same_frame, cross_frame = partition_from_manifest(dataset, manifest)
    print(
        f"  Train: {len(train_frame):,}  |  "
        f"Same-session test: {len(same_frame):,}  |  "
        f"Cross-session test: {len(cross_frame):,}"
    )

    # Verify session non-overlap — belt-and-suspenders.
    train_sessions = set(train_frame["session_id"])
    same_sessions = set(same_frame["session_id"])
    cross_sessions = set(cross_frame["session_id"])
    overlap = (train_sessions | same_sessions) & cross_sessions
    if overlap:
        raise EvaluationError(
            f"Cross-session test sessions appear in development data: {sorted(overlap)}. "
            "This indicates session leakage — check the manifest and dataset."
        )
    print(
        f"  Development sessions: {sorted(train_sessions | same_sessions)}\n"
        f"  Cross-session:        {sorted(cross_sessions)}"
    )

    print("\nClass distribution across all sessions:")
    print(_class_balance_report(dataset))
    _save_class_distribution(dataset, results_dir / "class_distribution.png")

    # -----------------------------------------------------------------
    # Per-representation evaluation.
    # -----------------------------------------------------------------
    all_metrics: list[dict[str, Any]] = []
    summary_parts: list[str] = []
    generalization_rows: list[dict[str, Any]] = []

    for representation in REPRESENTATIONS:
        print(f"\n{'-'*60}")
        print(f"  Representation: {representation.upper()}")
        print(f"{'-'*60}")

        pipeline = load_artifact(models_dir, representation)

        # Build feature matrices for the three partitions.
        X_same = build_feature_matrix(same_frame, representation)
        y_same = same_frame["gesture"].to_numpy(dtype=str)
        X_cross = build_feature_matrix(cross_frame, representation)
        y_cross = cross_frame["gesture"].to_numpy(dtype=str)

        # -- Same-session evaluation ----------------------------------
        y_same_pred = pipeline.predict(X_same)
        same_metrics = _compute_metrics(y_same, y_same_pred, GESTURE_LABELS, "same_session", representation)
        all_metrics.append(same_metrics)

        same_csv = results_dir / f"{representation}_same_session.csv"
        pd.DataFrame([same_metrics]).to_csv(same_csv, index=False)
        print(f"\n  Same-session test ({len(y_same):,} samples):")
        print(f"    Accuracy   : {same_metrics['accuracy']:.4f}")
        print(f"    Macro F1   : {same_metrics['macro_f1']:.4f}")
        print(f"    Macro Prec : {same_metrics['macro_precision']:.4f}")
        print(f"    Macro Rec  : {same_metrics['macro_recall']:.4f}")
        print("\n  Per-class report:")
        print(textwrap.indent(_full_report(y_same, y_same_pred, GESTURE_LABELS), "    "))

        _save_confusion_matrix(
            y_same,
            y_same_pred,
            GESTURE_LABELS,
            f"Confusion matrix - {representation} (same-session test)",
            results_dir / f"confusion_matrix_{representation}_same.png",
        )

        # -- Cross-session evaluation ---------------------------------
        y_cross_pred = pipeline.predict(X_cross)
        cross_metrics = _compute_metrics(y_cross, y_cross_pred, GESTURE_LABELS, "cross_session", representation)
        all_metrics.append(cross_metrics)

        cross_csv = results_dir / f"{representation}_cross_session.csv"
        pd.DataFrame([cross_metrics]).to_csv(cross_csv, index=False)
        print(f"\n  Cross-session test ({len(y_cross):,} samples):")
        print(f"    Accuracy   : {cross_metrics['accuracy']:.4f}")
        print(f"    Macro F1   : {cross_metrics['macro_f1']:.4f}")
        print(f"    Macro Prec : {cross_metrics['macro_precision']:.4f}")
        print(f"    Macro Rec  : {cross_metrics['macro_recall']:.4f}")
        print("\n  Per-class report:")
        print(textwrap.indent(_full_report(y_cross, y_cross_pred, GESTURE_LABELS), "    "))

        _save_confusion_matrix(
            y_cross,
            y_cross_pred,
            GESTURE_LABELS,
            f"Confusion matrix - {representation} (cross-session test)",
            results_dir / f"confusion_matrix_{representation}_cross.png",
        )

        generalization_rows.append(
            {
                "feature_representation": representation,
                "same_session_accuracy": same_metrics["accuracy"],
                "same_session_macro_f1": same_metrics["macro_f1"],
                "cross_session_accuracy": cross_metrics["accuracy"],
                "cross_session_macro_f1": cross_metrics["macro_f1"],
                "generalization_gap_accuracy": same_metrics["accuracy"] - cross_metrics["accuracy"],
                "generalization_gap_macro_f1": same_metrics["macro_f1"] - cross_metrics["macro_f1"],
            }
        )

    # -----------------------------------------------------------------
    # Four-cell generalization table (required by the brief).
    # -----------------------------------------------------------------
    generalization_frame = pd.DataFrame(generalization_rows)
    gen_csv = results_dir / "generalization_results.csv"
    generalization_frame.to_csv(gen_csv, index=False)
    print(f"\n{'='*70}")
    print("  REQUIRED FOUR-CELL GENERALIZATION TABLE")
    print(f"{'='*70}")
    header = f"  {'Feature Representation':<24} {'Same-Session Acc':>20} {'Cross-Session Acc':>20}"
    print(header)
    print("  " + "-" * 64)
    for _, row in generalization_frame.iterrows():
        label = str(row["feature_representation"]).replace("_", " ").title()
        print(f"  {label:<24} {row['same_session_accuracy']:>20.4f} {row['cross_session_accuracy']:>20.4f}")
    print(f"  Saved to: {gen_csv}")

    _save_generalization_comparison(generalization_frame, results_dir / "generalization_comparison.png")

    # -----------------------------------------------------------------
    # Combined CSV of all metrics.
    # -----------------------------------------------------------------
    all_metrics_frame = pd.DataFrame(all_metrics)
    all_metrics_frame.to_csv(results_dir / "all_evaluation_metrics.csv", index=False)

    # -----------------------------------------------------------------
    # Text summary derived entirely from measured results.
    # -----------------------------------------------------------------
    raw_row = generalization_frame.loc[generalization_frame["feature_representation"] == "raw"].iloc[0]
    inv_row = generalization_frame.loc[generalization_frame["feature_representation"] == "invariant"].iloc[0]

    def _fmt(value: float) -> str:
        return f"{value:.4f} ({value * 100:.2f}%)"

    gap_reduced = inv_row["generalization_gap_accuracy"] < raw_row["generalization_gap_accuracy"]
    gap_verdict = (
        "Invariant features reduce the generalization gap."
        if gap_reduced
        else "Raw coordinates achieve a smaller generalization gap in this experiment."
    )

    summary_text = f"""
GestureX evaluation summary
Generated from actual measured results -- not manually authored.
{'='*70}

Same-session accuracy
  Raw coordinates : {_fmt(raw_row['same_session_accuracy'])}
  Invariant feats : {_fmt(inv_row['same_session_accuracy'])}

Cross-session accuracy
  Raw coordinates : {_fmt(raw_row['cross_session_accuracy'])}
  Invariant feats : {_fmt(inv_row['cross_session_accuracy'])}

Generalization gap (same - cross, accuracy)
  Raw coordinates : {raw_row['generalization_gap_accuracy']:.4f}
  Invariant feats : {inv_row['generalization_gap_accuracy']:.4f}
  -> {gap_verdict}

Same-session macro F1
  Raw coordinates : {raw_row['same_session_macro_f1']:.4f}
  Invariant feats : {inv_row['same_session_macro_f1']:.4f}

Cross-session macro F1
  Raw coordinates : {raw_row['cross_session_macro_f1']:.4f}
  Invariant feats : {inv_row['cross_session_macro_f1']:.4f}

{'='*70}
Partition sizes (from manifest)
  Same-session train      : {len(train_frame):,}
  Same-session test       : {len(same_frame):,}
  Cross-session test      : {len(cross_frame):,}
  Development sessions    : {sorted(train_sessions | same_sessions)}
  Cross-session sessions  : {sorted(cross_sessions)}

Smoothing note
  Temporal smoothing (majority vote) is a deployment-only feature.
  It was NOT applied to any of the above metrics.

Leakage note
  The cross-session test sessions were not used during training or
  validation. The scalers inside the persisted pipelines were fitted
  only on same-session training rows.
{'='*70}
""".strip()

    summary_path = results_dir / "evaluation_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\n{summary_text}")
    print(f"\nFull summary saved to: {summary_path}")

    return {
        "generalization": generalization_frame.to_dict(orient="records"),
        "all_metrics": all_metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _default_data_path() -> Path:
    """Locate the processed dataset following the same convention as train.py."""

    from src.config import PROCESSED_DATASET_PATH
    return PROCESSED_DATASET_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GestureX offline evaluation: same-session and cross-session metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to the landmark CSV used during training.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MODELS_DIR / "split_manifest.json",
        help="Path to the split manifest written by train.py.",
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    data_path: Path = args.data or _default_data_path()

    try:
        evaluate(
            data_path=data_path,
            manifest_path=args.manifest,
            models_dir=args.models_dir,
            results_dir=args.results_dir,
        )
    except EvaluationError as exc:
        print(f"\nEvaluation error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
