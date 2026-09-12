"""MediaPipe landmark conversion, validation, detection, and drawing helpers.

The rest of GestureX works with a dependency-light ``(21, 3)`` NumPy array in
MediaPipe landmark order.  MediaPipe itself is imported only when a live
detector is constructed, which keeps offline feature tests usable without a
webcam or MediaPipe installation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Sequence

import numpy as np

try:  # Supports both ``python src/foo.py`` and ``import src.foo``.
    from .config import (
        LANDMARK_COUNT,
        MAX_NUM_HANDS,
        MIN_DETECTION_CONFIDENCE,
        MIN_TRACKING_CONFIDENCE,
        RAW_FEATURE_DIMENSION,
    )
except ImportError:  # pragma: no cover - exercised by direct script execution.
    from config import (  # type: ignore
        LANDMARK_COUNT,
        MAX_NUM_HANDS,
        MIN_DETECTION_CONFIDENCE,
        MIN_TRACKING_CONFIDENCE,
        RAW_FEATURE_DIMENSION,
    )


class LandmarkValidationError(ValueError):
    """Raised when a landmark object does not represent 21 finite XYZ points."""


@dataclass(frozen=True)
class HandDetection:
    """A validated result from :class:`HandLandmarkDetector`.

    ``raw_landmarks`` is retained solely for callers that want access to the
    original MediaPipe object.  ``landmarks`` is the stable project interface.
    ``handedness`` is ``'Left'``, ``'Right'``, or ``None`` when unknown.
    """

    landmarks: np.ndarray
    raw_landmarks: Any
    bounding_box: tuple[int, int, int, int]
    handedness: str | None = None


# Same topology as MediaPipe Hands.  Keeping it here avoids requiring
# MediaPipe just to render pre-recorded landmarks in a test or notebook.
HAND_CONNECTIONS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def _first_hand(candidate: Any) -> Any | None:
    """Return one hand from a MediaPipe result/container, if present."""

    if candidate is None:
        return None

    # A MediaPipe ``Hands.process`` result.
    if hasattr(candidate, "multi_hand_landmarks"):
        multi_hand_landmarks = candidate.multi_hand_landmarks
        return multi_hand_landmarks[0] if multi_hand_landmarks else None

    # A single NormalizedLandmarkList has a ``landmark`` field and is returned
    # unchanged.  A plain Python list should also be left unchanged.
    return candidate


def _all_hands(result: Any) -> list[tuple[Any, str | None]]:
    """Return all detected hands from a MediaPipe result as (landmarks, handedness) pairs.

    Handedness is ``'Left'`` or ``'Right'`` as reported by MediaPipe (note:
    MediaPipe returns handedness from the *camera's* perspective, which is the
    mirror of the *person's* perspective when a mirror-flip is applied in the
    app).  Returns an empty list when no hands are detected.
    """

    if result is None or not hasattr(result, "multi_hand_landmarks"):
        return []
    multi_landmarks = result.multi_hand_landmarks
    if not multi_landmarks:
        return []
    multi_handedness = getattr(result, "multi_handedness", None) or []
    pairs: list[tuple[Any, str | None]] = []
    for index, hand_landmarks in enumerate(multi_landmarks):
        label: str | None = None
        if index < len(multi_handedness):
            classifications = getattr(multi_handedness[index], "classification", [])
            if classifications:
                label = classifications[0].label  # 'Left' or 'Right'
        pairs.append((hand_landmarks, label))
    return pairs


def validate_landmarks(landmarks: Any) -> np.ndarray:
    """Convert a landmark sequence into a finite ``float64`` array of shape ``(21, 3)``.

    Accepted input includes MediaPipe ``NormalizedLandmarkList`` objects,
    sequences of objects exposing ``x``, ``y``, and ``z``, and numeric arrays.
    This deliberately rejects flattened vectors: callers should use
    :func:`unflatten_landmarks` when the source is a stored 63-D row.
    """

    if landmarks is None:
        raise LandmarkValidationError("Landmarks are missing; expected 21 XYZ points.")

    point_sequence = getattr(landmarks, "landmark", landmarks)
    try:
        points = list(point_sequence)
    except TypeError as exc:
        raise LandmarkValidationError(
            "Landmarks must be a sequence or a MediaPipe landmark list."
        ) from exc

    if len(points) != LANDMARK_COUNT:
        raise LandmarkValidationError(
            f"Expected {LANDMARK_COUNT} landmarks, received {len(points)}."
        )

    normalized_rows: list[list[float]] = []
    for index, point in enumerate(points):
        if all(hasattr(point, axis) for axis in ("x", "y", "z")):
            row = [point.x, point.y, point.z]
        else:
            try:
                row = list(point)
            except TypeError as exc:
                raise LandmarkValidationError(
                    f"Landmark {index} must expose x/y/z coordinates."
                ) from exc
            if len(row) != 3:
                raise LandmarkValidationError(
                    f"Landmark {index} has {len(row)} values; expected exactly 3 (x, y, z)."
                )
        normalized_rows.append(row)

    try:
        array = np.asarray(normalized_rows, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LandmarkValidationError("Landmark coordinates must be numeric.") from exc

    if array.shape != (LANDMARK_COUNT, 3):  # Defensive; row validation above should catch it.
        raise LandmarkValidationError(
            f"Expected landmark array shape ({LANDMARK_COUNT}, 3), got {array.shape}."
        )
    if not np.isfinite(array).all():
        raise LandmarkValidationError("Landmark coordinates contain NaN or infinite values.")
    return array


def extract_landmarks(hand_landmarks: Any) -> np.ndarray | None:
    """Extract the first detected hand as a validated ``(21, 3)`` array.

    ``None`` and MediaPipe results with no detected hands return ``None``.  A
    malformed *present* hand raises :class:`LandmarkValidationError` instead of
    becoming a deceptively valid training sample.
    """

    first_hand = _first_hand(hand_landmarks)
    if first_hand is None:
        return None
    return validate_landmarks(first_hand)


def unflatten_landmarks(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Restore a 63-D raw feature vector to the project ``(21, 3)`` layout."""

    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LandmarkValidationError("Flattened landmarks must contain numeric values.") from exc

    if vector.size != RAW_FEATURE_DIMENSION:
        raise LandmarkValidationError(
            f"Expected {RAW_FEATURE_DIMENSION} flattened coordinates, got {vector.size}."
        )
    reshaped = vector.reshape(LANDMARK_COUNT, 3)
    if not np.isfinite(reshaped).all():
        raise LandmarkValidationError("Flattened landmarks contain NaN or infinite values.")
    return reshaped


def flatten_landmarks(landmarks: Any) -> np.ndarray:
    """Return raw coordinates in the documented ``x0,y0,z0,...,x20,y20,z20`` order."""

    return validate_landmarks(landmarks).reshape(RAW_FEATURE_DIMENSION).copy()


def landmark_pixel_coordinates(
    landmarks: Any,
    frame_shape: Sequence[int],
) -> np.ndarray:
    """Convert normalized MediaPipe XY coordinates to clipped integer pixels."""

    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain at least height and width.")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid frame dimensions: height={height}, width={width}.")

    points = validate_landmarks(landmarks)
    pixels = np.empty((LANDMARK_COUNT, 2), dtype=np.int32)
    pixels[:, 0] = np.clip(np.rint(points[:, 0] * (width - 1)), 0, width - 1).astype(
        np.int32
    )
    pixels[:, 1] = np.clip(np.rint(points[:, 1] * (height - 1)), 0, height - 1).astype(
        np.int32
    )
    return pixels


def landmarks_to_bbox(
    landmarks: Any,
    frame_shape: Sequence[int],
    *,
    padding_fraction: float = 0.05,
) -> tuple[int, int, int, int]:
    """Return a clipped ``(x_min, y_min, x_max, y_max)`` pixel bounding box."""

    if padding_fraction < 0:
        raise ValueError("padding_fraction must be non-negative.")
    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain at least height and width.")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    pixels = landmark_pixel_coordinates(landmarks, frame_shape)
    horizontal_padding = int(round(width * padding_fraction))
    vertical_padding = int(round(height * padding_fraction))

    x_min = max(0, int(pixels[:, 0].min()) - horizontal_padding)
    y_min = max(0, int(pixels[:, 1].min()) - vertical_padding)
    x_max = min(width - 1, int(pixels[:, 0].max()) + horizontal_padding)
    y_max = min(height - 1, int(pixels[:, 1].max()) + vertical_padding)
    return x_min, y_min, x_max, y_max


# Clear aliases used by scripts and by readers familiar with either naming.
bounding_box_from_landmarks = landmarks_to_bbox
get_bounding_box = landmarks_to_bbox


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment-dependent.
        raise RuntimeError(
            "OpenCV is required for drawing or webcam capture. Install requirements.txt."
        ) from exc
    return cv2


def draw_hand_annotations(
    frame: np.ndarray,
    landmarks: Any,
    *,
    bounding_box: tuple[int, int, int, int] | None = None,
    label: str | None = None,
    confidence: float | None = None,
    draw_connections: bool = True,
) -> np.ndarray:
    """Draw landmark topology, a box, and optional prediction text onto ``frame``.

    The function mutates and also returns ``frame`` to mirror OpenCV's usual
    drawing style.  It accepts both raw MediaPipe landmarks and validated arrays.
    """

    if not isinstance(frame, np.ndarray) or frame.ndim < 2:
        raise ValueError("frame must be a NumPy image array with height and width.")
    cv2 = _require_cv2()
    pixels = landmark_pixel_coordinates(landmarks, frame.shape)

    if draw_connections:
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                tuple(pixels[start]),
                tuple(pixels[end]),
                (80, 220, 80),
                2,
                cv2.LINE_AA,
            )
    for x_coord, y_coord in pixels:
        cv2.circle(frame, (int(x_coord), int(y_coord)), 3, (40, 80, 255), -1, cv2.LINE_AA)

    box = bounding_box or landmarks_to_bbox(landmarks, frame.shape)
    x_min, y_min, x_max, y_max = box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 180, 0), 2, cv2.LINE_AA)

    if label:
        text = label
        if confidence is not None and np.isfinite(confidence):
            text = f"{label}: {confidence:.1%}"
        text_origin = (x_min, max(25, y_min - 10))
        cv2.putText(
            frame,
            text,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


draw_landmarks = draw_hand_annotations


class HandLandmarkDetector:
    """Small context-managed wrapper around MediaPipe Hands for one-hand use.

    The detector performs BGR-to-RGB conversion, selects the first detected
    hand, and returns a :class:`HandDetection`.  It is intentionally separate
    from feature extraction so landmark latency can be benchmarked independently.
    """

    def __init__(
        self,
        *,
        static_image_mode: bool = False,
        max_num_hands: int = MAX_NUM_HANDS,
        model_complexity: int = 1,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ) -> None:
        if max_num_hands < 1:
            raise ValueError("max_num_hands must be at least 1.")
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - environment-dependent.
            raise RuntimeError(
                "MediaPipe is required for live landmark detection. "
                "Install the project requirements first."
            ) from exc

        self._cv2 = _require_cv2()
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._closed = False

    def detect(self, frame: np.ndarray) -> HandDetection | None:
        """Run MediaPipe on a BGR frame and return one hand, if present."""

        if self._closed:
            raise RuntimeError("HandLandmarkDetector is closed.")
        if not isinstance(frame, np.ndarray) or frame.ndim < 3:
            raise ValueError("frame must be a color NumPy image array (H, W, C).")

        rgb_frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb_frame)
        raw_hand = _first_hand(result)
        if raw_hand is None:
            return None
        points = validate_landmarks(raw_hand)
        return HandDetection(
            landmarks=points,
            raw_landmarks=raw_hand,
            bounding_box=landmarks_to_bbox(points, frame.shape),
        )

    def detect_all(self, frame: np.ndarray) -> list[HandDetection]:
        """Run MediaPipe and return a :class:`HandDetection` for every detected hand.

        Returns an empty list when no hands are found.  Handedness (``'Left'`` /
        ``'Right'``) is populated from the MediaPipe classification.
        """

        if self._closed:
            raise RuntimeError("HandLandmarkDetector is closed.")
        if not isinstance(frame, np.ndarray) or frame.ndim < 3:
            raise ValueError("frame must be a color NumPy image array (H, W, C).")

        rgb_frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb_frame)
        detections: list[HandDetection] = []
        for raw_hand, handedness in _all_hands(result):
            try:
                points = validate_landmarks(raw_hand)
            except LandmarkValidationError:
                continue
            detections.append(
                HandDetection(
                    landmarks=points,
                    raw_landmarks=raw_hand,
                    bounding_box=landmarks_to_bbox(points, frame.shape),
                    handedness=handedness,
                )
            )
        return detections

    process = detect

    def close(self) -> None:
        """Release MediaPipe resources. Safe to call more than once."""

        if not self._closed:
            self._hands.close()
            self._closed = True

    def __enter__(self) -> "HandLandmarkDetector":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
