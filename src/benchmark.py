"""Repeated latency benchmarking for the GestureX inference pipeline.

Measures and reports separately:

1. **MediaPipe landmark detection** — BGR frame → validated (21, 3) array.
2. **Feature extraction** — validated landmarks → feature vector (raw or invariant).
3. **Classifier inference** — feature vector → class label and probability.
4. **End-to-end pipeline** — all of the above in one call, approximating the
   per-frame cost excluding operating-system camera I/O.

Results are written to ``results/latency_results.csv`` with mean, median,
and standard deviation (in milliseconds) and FPS where appropriate.

Why separate them?
------------------
*Classifier inference latency* tells you the cost of the ML model alone.
*End-to-end pipeline latency / FPS* is what the user experiences in the live
app.  They differ because MediaPipe, memory allocation, and Python overhead
all contribute to the latter but not the former.  Reporting only one figure
would be misleading.

The benchmark uses pre-recorded landmark arrays (randomly generated valid
samples) rather than requiring a webcam, so it runs reproducibly in any
environment that has the trained models.  If a trained model is not found the
classifier timing step is skipped and the report explains this.

Usage::

    python src/benchmark.py                     # uses config defaults
    python src/benchmark.py --repetitions 300  # increase sample count
    python src/benchmark.py --no-webcam        # skip MediaPipe timing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Repository bootstrap.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DEFAULT_BENCHMARK_REPETITIONS,
    LANDMARK_COUNT,
    MODELS_DIR,
    NORMALIZATION_EPSILON,
    RESULTS_DIR,
)
from src.features import extract_features, extract_invariant_features, extract_raw_features  # noqa: E402
from src.landmarks import HandLandmarkDetector, flatten_landmarks, validate_landmarks  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic landmark generation for reproducible timing.
# ---------------------------------------------------------------------------
def _random_valid_landmarks(rng: np.random.Generator) -> np.ndarray:
    """Generate a syntactically valid (21, 3) landmark array.

    Points are uniformly drawn from [0, 1] for x/y and [-0.1, 0.1] for z,
    matching the approximate range of normalized MediaPipe coordinates.
    The wrist-to-middle-MCP distance is checked to exceed the normalization
    epsilon so invariant features can always be extracted.
    """

    for _ in range(100):
        points = np.empty((LANDMARK_COUNT, 3), dtype=np.float64)
        points[:, :2] = rng.uniform(0.05, 0.95, size=(LANDMARK_COUNT, 2))
        points[:, 2] = rng.uniform(-0.1, 0.1, size=LANDMARK_COUNT)
        # Landmark 9 is middle-finger MCP; landmark 0 is wrist.
        if np.linalg.norm(points[9] - points[0]) > NORMALIZATION_EPSILON * 10:
            return points
    # Fallback: a simple spread that always satisfies the scale guard.
    points = np.zeros((LANDMARK_COUNT, 3), dtype=np.float64)
    for index in range(LANDMARK_COUNT):
        points[index] = [index * 0.05, index * 0.03, 0.0]
    return points


# ---------------------------------------------------------------------------
# Timing utilities.
# ---------------------------------------------------------------------------
def _time_repeated(
    func: Callable[[], object],
    repetitions: int,
    warmup: int = 5,
) -> np.ndarray:
    """Return an array of per-call durations in milliseconds.

    The first ``warmup`` calls are discarded to exclude JIT warm-up
    and MediaPipe model-loading costs from the reported statistics.
    """

    for _ in range(warmup):
        func()

    times_ms = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times_ms[index] = (end - start) * 1000.0
    return times_ms


def _stats(times_ms: np.ndarray) -> dict[str, float]:
    """Compute summary statistics from a millisecond timing array."""

    return {
        "mean_ms": float(np.mean(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "std_ms": float(np.std(times_ms, ddof=1)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "fps": float(1000.0 / np.mean(times_ms)) if np.mean(times_ms) > 0 else float("inf"),
        "n_repetitions": int(len(times_ms)),
    }


def _print_stats(label: str, stats: dict[str, float]) -> None:
    print(f"  {label}")
    print(f"    Mean   : {stats['mean_ms']:.3f} ms  (FPS ≈ {stats['fps']:.1f})")
    print(f"    Median : {stats['median_ms']:.3f} ms")
    print(f"    Std    : {stats['std_ms']:.3f} ms")
    print(f"    Range  : [{stats['min_ms']:.3f}, {stats['max_ms']:.3f}] ms")
    print(f"    N      : {stats['n_repetitions']}")
    print()


# ---------------------------------------------------------------------------
# Individual benchmark stages.
# ---------------------------------------------------------------------------
def _benchmark_mediapipe(
    repetitions: int,
    camera_index: int,
) -> dict[str, float] | None:
    """Time MediaPipe landmark detection on real webcam frames.

    Returns None if the camera cannot be opened; the script continues
    without crashing so other stages can still be measured.
    """

    try:
        import cv2
    except ImportError:
        print("  [SKIP] OpenCV not available; skipping MediaPipe timing.")
        return None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(
            f"  [SKIP] Camera {camera_index} not available; "
            "skipping MediaPipe detection timing. "
            "Feature-extraction and classifier timing will still run."
        )
        cap.release()
        return None

    frames: list[np.ndarray] = []
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("  [SKIP] Webcam returned an empty frame; skipping MediaPipe timing.")
        return None

    # Replicate the single frame for repeated timing without repeated I/O.
    frames = [frame.copy() for _ in range(repetitions + 20)]

    with HandLandmarkDetector() as detector:
        frame_iter = iter(frames)

        def _detect_one() -> None:
            detector.detect(next(frame_iter))  # type: ignore[arg-type]

        # Pre-run enough frames for warmup before measuring.
        warmup_frames = [frame.copy() for _ in range(10)]
        for wf in warmup_frames:
            detector.detect(wf)

        times_ms = np.empty(repetitions, dtype=np.float64)
        detection_frames = [frame.copy() for _ in range(repetitions)]
        for index, f in enumerate(detection_frames):
            start = time.perf_counter()
            detector.detect(f)
            end = time.perf_counter()
            times_ms[index] = (end - start) * 1000.0

    return _stats(times_ms)


def _benchmark_feature_extraction(
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """Time raw and invariant feature extraction on synthetic landmarks."""

    results: dict[str, dict[str, float]] = {}
    for representation in ("raw", "invariant"):
        landmarks_batch = [_random_valid_landmarks(rng) for _ in range(repetitions + 10)]
        batch_iter = iter(landmarks_batch)

        def _extract() -> None:
            pts = next(batch_iter)  # type: ignore[assignment]
            extract_features(pts, representation)  # type: ignore[arg-type]

        # Warmup.
        for _ in range(5):
            extract_features(_random_valid_landmarks(rng), representation)

        times_ms = np.empty(repetitions, dtype=np.float64)
        timed_batch = [_random_valid_landmarks(rng) for _ in range(repetitions)]
        for index, pts in enumerate(timed_batch):
            start = time.perf_counter()
            extract_features(pts, representation)
            end = time.perf_counter()
            times_ms[index] = (end - start) * 1000.0

        results[representation] = _stats(times_ms)
    return results


def _benchmark_classifier(
    repetitions: int,
    models_dir: Path,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """Time classifier-only inference (feature vector → prediction) per representation."""

    results: dict[str, dict[str, float]] = {}
    for representation in ("raw", "invariant"):
        artifact_path = models_dir / f"gesturex_{representation}_best.joblib"
        if not artifact_path.exists():
            print(
                f"  [SKIP] No trained model at {artifact_path}; skipping {representation} "
                "classifier timing. Run ``python src/train.py`` first."
            )
            continue

        artifact = joblib.load(artifact_path)
        pipeline = artifact.get("pipeline") if isinstance(artifact, dict) else artifact
        if pipeline is None:
            print(f"  [SKIP] Artifact at {artifact_path} has no usable pipeline.")
            continue

        dim = 63 if representation == "raw" else 8
        vectors = [rng.standard_normal(dim).reshape(1, -1) for _ in range(repetitions + 10)]
        # Warmup.
        for vec in vectors[:5]:
            pipeline.predict(vec)

        times_ms = np.empty(repetitions, dtype=np.float64)
        for index in range(repetitions):
            vec = vectors[index + 5]
            start = time.perf_counter()
            pipeline.predict(vec)
            end = time.perf_counter()
            times_ms[index] = (end - start) * 1000.0

        results[representation] = _stats(times_ms)
    return results


def _benchmark_end_to_end(
    repetitions: int,
    models_dir: Path,
    rng: np.random.Generator,
    camera_index: int,
) -> dict[str, float] | None:
    """Time the complete per-frame pipeline: detection → features → predict.

    Requires a working camera.  Returns None if unavailable.
    """

    try:
        import cv2
    except ImportError:
        return None

    # Pick whichever model exists (prefer best artifact).
    best_path = models_dir / "gesturex_best.joblib"
    fallback_path = models_dir / "gesturex_raw_best.joblib"
    artifact_path = best_path if best_path.exists() else (
        fallback_path if fallback_path.exists() else None
    )
    if artifact_path is None:
        print("  [SKIP] No trained model found; skipping end-to-end timing.")
        return None

    artifact = joblib.load(artifact_path)
    if isinstance(artifact, dict):
        pipeline = artifact.get("pipeline")
        representation = artifact.get("feature_representation", "raw")
    else:
        pipeline = artifact
        representation = "raw"
    if pipeline is None:
        return None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        return None

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None

    # Replicate the frame to avoid camera I/O in the timed section.
    cached_frames = [frame.copy() for _ in range(repetitions + 10)]

    times_ms = np.empty(repetitions, dtype=np.float64)
    with HandLandmarkDetector() as detector:
        # Warmup.
        for wf in cached_frames[:5]:
            det = detector.detect(wf)
            if det is not None:
                vec = extract_features(det.landmarks, representation)
                pipeline.predict(vec.reshape(1, -1))

        for index in range(repetitions):
            f = cached_frames[index + 5]
            start = time.perf_counter()
            det = detector.detect(f)
            if det is not None:
                vec = extract_features(det.landmarks, representation)
                pipeline.predict(vec.reshape(1, -1))
            end = time.perf_counter()
            times_ms[index] = (end - start) * 1000.0

    return _stats(times_ms)


# ---------------------------------------------------------------------------
# Main benchmark routine.
# ---------------------------------------------------------------------------
def run_benchmark(
    *,
    repetitions: int,
    models_dir: Path,
    results_dir: Path,
    camera_index: int,
    skip_mediapipe: bool,
    random_seed: int,
) -> None:
    """Execute all benchmark stages and write ``latency_results.csv``."""

    rng = np.random.default_rng(random_seed)
    results_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    print("=" * 70)
    print("GestureX latency benchmark")
    print("=" * 70)
    print(f"\nRepetitions per stage : {repetitions}")
    print(f"Random seed           : {random_seed}")
    print(f"Models directory      : {models_dir}")
    print(f"Results directory     : {results_dir}\n")

    # ── Stage 1: MediaPipe detection ──────────────────────────────────────
    print("Stage 1 — MediaPipe landmark detection latency")
    if skip_mediapipe:
        print("  [SKIP] --no-webcam flag set.")
        mp_stats = None
    else:
        mp_stats = _benchmark_mediapipe(repetitions, camera_index)
    if mp_stats:
        _print_stats("MediaPipe detection (BGR frame → 21 landmarks)", mp_stats)
        records.append({"stage": "mediapipe_detection", **mp_stats})
    else:
        print()

    # ── Stage 2: Feature extraction ───────────────────────────────────────
    print("Stage 2 — Feature extraction latency (synthetic landmarks)")
    feat_stats = _benchmark_feature_extraction(repetitions, rng)
    for rep, stats in feat_stats.items():
        _print_stats(f"Feature extraction ({rep})", stats)
        records.append({"stage": f"feature_extraction_{rep}", **stats})

    # ── Stage 3: Classifier inference ─────────────────────────────────────
    print("Stage 3 — Classifier inference latency (feature vector → prediction)")
    print(
        "  Note: this measures the scikit-learn pipeline only, not MediaPipe or feature extraction.\n"
        "  This is the 'classifier inference latency' discussed in the README.\n"
    )
    clf_stats = _benchmark_classifier(repetitions, models_dir, rng)
    for rep, stats in clf_stats.items():
        _print_stats(f"Classifier inference ({rep})", stats)
        records.append({"stage": f"classifier_inference_{rep}", **stats})

    # ── Stage 4: End-to-end pipeline ──────────────────────────────────────
    print("Stage 4 — End-to-end pipeline latency (detection + features + predict)")
    print(
        "  Note: this includes MediaPipe, feature extraction, and classifier inference.\n"
        "  This reflects what the deployment loop costs per frame.\n"
    )
    if skip_mediapipe:
        print("  [SKIP] --no-webcam flag set; end-to-end timing skipped.\n")
        e2e_stats = None
    else:
        e2e_stats = _benchmark_end_to_end(repetitions, models_dir, rng, camera_index)
    if e2e_stats:
        _print_stats("End-to-end pipeline (detect + extract + predict)", e2e_stats)
        records.append({"stage": "end_to_end_pipeline", **e2e_stats})
    else:
        print()

    # ── Summary ──────────────────────────────────────────────────────────
    if records:
        output_path = results_dir / "latency_results.csv"
        pd.DataFrame(records).to_csv(output_path, index=False)
        print(f"Benchmark complete. Results saved to: {output_path}")

        print("\n" + "=" * 70)
        print("LATENCY SUMMARY")
        print("=" * 70)
        for record in records:
            stage = str(record["stage"]).replace("_", " ").title()
            mean_ms = float(record.get("mean_ms", float("nan")))
            fps = float(record.get("fps", float("nan")))
            print(f"  {stage:<48} {mean_ms:>8.3f} ms  ({fps:.1f} FPS)")
        print("=" * 70)
        print(
            "\nKey distinction:\n"
            "  • Classifier inference: cost of the sklearn model alone (stage 3).\n"
            "  • End-to-end FPS      : what the live deployment loop achieves (stage 4).\n"
            "  Reporting only classifier FPS would overstate real-time performance."
        )
    else:
        print("No benchmark stages completed. Check that models and camera are available.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GestureX latency benchmark — repeated measurements per pipeline stage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_BENCHMARK_REPETITIONS,
        help="Number of repeated measurements per stage (after warmup).",
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for MediaPipe and end-to-end timing.",
    )
    parser.add_argument(
        "--no-webcam",
        dest="skip_mediapipe",
        action="store_true",
        help="Skip stages that require a camera; benchmark feature extraction and classifiers only.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic landmark generation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repetitions < 10:
        parser.error("--repetitions should be at least 10 for meaningful statistics.")
    run_benchmark(
        repetitions=args.repetitions,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        camera_index=args.camera,
        skip_mediapipe=args.skip_mediapipe,
        random_seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
