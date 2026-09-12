"""Run GestureX real-time hand gesture recognition from a webcam.

The application deliberately keeps temporal smoothing in this deployment
layer.  Offline metrics are always computed from individual frames, so a
video-specific majority vote cannot inflate reported classifier performance.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Iterable

import cv2
import joblib
import numpy as np


# Allow ``python deployment/app.py`` from either the repository root or a
# different current working directory, as documented in the README.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_SMOOTHING_WINDOW,
    GESTURE_DISPLAY_NAMES,
    MODEL_ARTIFACT_PATH,
    MODEL_METADATA_PATH,
    RAW_REPRESENTATION,
    SUPPORTED_REPRESENTATIONS,
)
from src.features import extract_features  # noqa: E402
from src.landmarks import HandLandmarkDetector, draw_hand_annotations  # noqa: E402


UNKNOWN_LABEL = "UNKNOWN"


@dataclass(frozen=True)
class LoadedArtifact:
    """The model plus the metadata needed to turn a frame into a prediction."""

    pipeline: Any
    representation: str
    labels: tuple[str, ...]
    model_name: str


class PredictionSmoother:
    """Deployment-only majority-vote smoother with deterministic tie-breaking."""

    def __init__(self, window_size: int) -> None:
        if window_size < 1:
            raise ValueError("Smoothing window must be at least 1.")
        self._history: Deque[str] = deque(maxlen=window_size)

    def add(self, label: str) -> str:
        """Add an accepted prediction and return its recent majority vote.

        Ties are resolved by the newest matching label.  That keeps the UI
        responsive when the hand deliberately changes gesture.
        """

        self._history.append(label)
        counts = Counter(self._history)
        highest_count = max(counts.values())
        for candidate in reversed(self._history):
            if counts[candidate] == highest_count:
                return candidate
        return label  # unreachable, but makes the return contract explicit

    def clear(self) -> None:
        """Forget old labels when the hand leaves the frame."""

        self._history.clear()


def _read_metadata(metadata_path: Path) -> dict[str, Any]:
    """Read optional JSON metadata, failing clearly if it is malformed."""

    if not metadata_path.exists():
        return {}
    try:
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model metadata is not valid JSON: {metadata_path}") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Model metadata must be a JSON object: {metadata_path}")
    return loaded


def _first_present(mapping: dict[str, Any], names: Iterable[str]) -> Any | None:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def load_artifact(model_path: Path, metadata_path: Path) -> LoadedArtifact:
    """Load the validation-selected model and validate its deployment contract.

    ``train.py`` stores a small dictionary artifact.  A direct scikit-learn
    pipeline is also accepted so users can migrate an older local model after
    providing its representation in ``gesturex_metadata.json``.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run `python src/train.py` "
            "after collecting at least two recording groups."
        )

    loaded = joblib.load(model_path)
    metadata = _read_metadata(metadata_path)
    artifact: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
    pipeline = _first_present(artifact, ("pipeline", "model", "estimator"))
    if pipeline is None:
        pipeline = loaded

    representation = _first_present(
        artifact, ("feature_representation", "representation")
    )
    if representation is None:
        representation = _first_present(
            metadata, ("feature_representation", "representation")
        )
    if representation is None:
        raise RuntimeError(
            "The model artifact does not state its feature representation. "
            "Re-run `python src/train.py` to create a GestureX deployment artifact."
        )
    representation = str(representation).strip().lower()
    if representation not in SUPPORTED_REPRESENTATIONS:
        raise RuntimeError(
            f"Model uses unsupported representation {representation!r}; expected one "
            f"of {', '.join(SUPPORTED_REPRESENTATIONS)}."
        )

    labels = _first_present(artifact, ("class_labels", "labels", "classes"))
    if labels is None:
        labels = _first_present(metadata, ("class_labels", "labels", "classes"))
    if labels is None and hasattr(pipeline, "classes_"):
        labels = pipeline.classes_
    if labels is None:
        raise RuntimeError("The saved model does not include its class labels.")
    normalized_labels = tuple(str(label) for label in labels)
    if not normalized_labels:
        raise RuntimeError("The saved model contains an empty class-label list.")

    model_name = str(
        _first_present(artifact, ("model_name", "selected_model", "best_model_name"))
        or _first_present(metadata, ("model_name", "selected_model", "best_model_name"))
        or type(pipeline).__name__
    )
    return LoadedArtifact(pipeline, representation, normalized_labels, model_name)


def predict_one(artifact: LoadedArtifact, feature_vector: np.ndarray) -> tuple[str, float | None]:
    """Return the predicted label and probability when the estimator supports it."""

    matrix = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
    label = str(artifact.pipeline.predict(matrix)[0])
    if not hasattr(artifact.pipeline, "predict_proba"):
        return label, None

    probabilities = np.asarray(artifact.pipeline.predict_proba(matrix)[0], dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise RuntimeError("Model returned malformed class probabilities.")
    return label, float(np.max(probabilities))


def _display_label(label: str) -> str:
    return GESTURE_DISPLAY_NAMES.get(label, label.replace("_", " ").title())


def draw_hud(
    frame: np.ndarray,
    hand_results: list[tuple[str, float | None, str | None]],
    fps: float,
    model_name: str,
) -> None:
    """Draw a status panel showing predictions for up to two hands."""

    num_hands = len(hand_results)
    panel_height = 70 + num_hands * 38 if num_hands else 108
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (420, panel_height), (15, 20, 25), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    y = 30
    if not hand_results:
        cv2.putText(
            frame,
            "No hand detected",
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 191, 255),
            2,
            cv2.LINE_AA,
        )
        y += 34
    else:
        for label, confidence, side in hand_results:
            side_tag = f"[{side}] " if side else ""
            display = f"{side_tag}{_display_label(label)}"
            if confidence is not None:
                display += f"  {confidence * 100:.0f}%"
            colour = (50, 205, 50) if label != UNKNOWN_LABEL else (0, 191, 255)
            cv2.putText(
                frame,
                display,
                (14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                colour,
                2,
                cv2.LINE_AA,
            )
            y += 34

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}  |  Model: {model_name}",
        (14, y + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Press Q or Esc to quit",
        (14, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.49,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Background inference worker
# ---------------------------------------------------------------------------
@dataclass
class _InferenceState:
    """Shared state between the capture loop and the inference thread."""
    latest_frame: np.ndarray | None = field(default=None)
    latest_detections: list = field(default_factory=list)
    latest_results: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    frame_ready: threading.Event = field(default_factory=threading.Event)


class InferenceWorker(threading.Thread):
    """Runs MediaPipe detection + sklearn inference in a background thread.

    The main loop deposits the latest camera frame; this thread picks it up,
    runs the expensive work, and stores the result.  The display loop reads
    results without ever blocking on inference.
    """

    def __init__(
        self,
        state: "_InferenceState",
        artifact: "LoadedArtifact",
        smoothers: dict,
        confidence_threshold: float,
        mirror: bool,
    ) -> None:
        super().__init__(daemon=True, name="InferenceWorker")
        self._state = state
        self._artifact = artifact
        self._smoothers = smoothers
        self._threshold = confidence_threshold
        self._mirror = mirror

    @staticmethod
    def _display_side(mp_side: str | None, mirrored: bool) -> str | None:
        if mp_side is None:
            return None
        return mp_side

    def run(self) -> None:
        import mediapipe as mp
        import cv2 as _cv2
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,           # lite model — much faster
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        try:
            while not self._state.stop_event.is_set():
                # Wait for a new frame (up to 30 ms so we can check stop_event).
                if not self._state.frame_ready.wait(timeout=0.03):
                    continue
                self._state.frame_ready.clear()

                with self._state.lock:
                    frame = self._state.latest_frame
                if frame is None:
                    continue

                rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                from src.landmarks import (
                    _all_hands, validate_landmarks, landmarks_to_bbox,
                    LandmarkValidationError, HandDetection,
                )
                detections = []
                for raw_hand, handedness in _all_hands(result):
                    try:
                        pts = validate_landmarks(raw_hand)
                    except LandmarkValidationError:
                        continue
                    detections.append(HandDetection(
                        landmarks=pts,
                        raw_landmarks=raw_hand,
                        bounding_box=landmarks_to_bbox(pts, frame.shape),
                        handedness=handedness,
                    ))

                results = []
                active_keys: set[str] = set()
                for det in detections:
                    mp_side = det.handedness
                    display_side = self._display_side(mp_side, self._mirror)
                    smoother_key = mp_side if mp_side in self._smoothers else "Unknown"
                    active_keys.add(smoother_key)
                    try:
                        fvec = extract_features(det.landmarks, self._artifact.representation)
                        label, conf = predict_one(self._artifact, fvec)
                    except Exception:
                        label, conf = UNKNOWN_LABEL, None
                    if conf is None or conf >= self._threshold:
                        displayed = self._smoothers[smoother_key].add(label)
                    else:
                        displayed = UNKNOWN_LABEL
                    results.append((displayed, conf, display_side, det))

                for key, smoother in self._smoothers.items():
                    if key not in active_keys:
                        smoother.clear()

                with self._state.lock:
                    self._state.latest_detections = detections
                    self._state.latest_results = results
        finally:
            hands.close()


def run(args: argparse.Namespace) -> None:
    """Open the webcam and render per-hand predictions with a threaded inference worker."""

    artifact = load_artifact(Path(args.model), Path(args.metadata))
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. Try --camera 1, "
            "check camera permissions, or close another webcam application."
        )

    # Reduce camera buffer to 1 so we always read the freshest frame.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    smoothers: dict[str, PredictionSmoother] = {
        "Left": PredictionSmoother(args.smoothing_window),
        "Right": PredictionSmoother(args.smoothing_window),
        "Unknown": PredictionSmoother(args.smoothing_window),
    }

    state = _InferenceState()
    worker = InferenceWorker(
        state, artifact, smoothers,
        confidence_threshold=args.confidence_threshold,
        mirror=args.mirror,
    )
    worker.start()
    previous_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Webcam returned an empty frame.")
            if args.mirror:
                frame = cv2.flip(frame, 1)

            # Send frame to inference worker (non-blocking).
            with state.lock:
                state.latest_frame = frame.copy()
            state.frame_ready.set()

            # Read latest result (never blocks).
            with state.lock:
                inference_results = list(state.latest_results)

            hud_results: list[tuple[str, float | None, str | None]] = []
            for label, confidence, display_side, detection in inference_results:
                hud_results.append((label, confidence, display_side))

                draw_hand_annotations(
                    frame,
                    detection.landmarks,
                    bounding_box=detection.bounding_box,
                    label=None,
                    confidence=None,
                )
                x_min, y_min, _, _ = detection.bounding_box
                side_tag = f"{display_side}: " if display_side else ""
                hand_text = f"{side_tag}{_display_label(label)}"
                cv2.putText(
                    frame, hand_text,
                    (x_min, max(20, y_min - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-9)
            previous_time = now
            draw_hud(frame, hud_results, fps, artifact.model_name)
            cv2.imshow("GestureX - Real-Time Hand Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        state.stop_event.set()
        worker.join(timeout=2.0)
        capture.release()
        cv2.destroyAllWindows()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a validation-selected GestureX model from a webcam."
    )
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--model", type=Path, default=MODEL_ARTIFACT_PATH)
    parser.add_argument("--metadata", type=Path, default=MODEL_METADATA_PATH)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Show UNKNOWN below this probability (0 through 1).",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=DEFAULT_SMOOTHING_WINDOW,
        help="Number of accepted predictions used for deployment-only voting.",
    )
    parser.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        help="Do not mirror the webcam preview.",
    )
    parser.set_defaults(mirror=True)
    arguments = parser.parse_args()
    if not 0.0 <= arguments.confidence_threshold <= 1.0:
        parser.error("--confidence-threshold must be between 0 and 1.")
    if arguments.smoothing_window < 1:
        parser.error("--smoothing-window must be at least 1.")
    return arguments


if __name__ == "__main__":
    try:
        run(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"GestureX deployment error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
