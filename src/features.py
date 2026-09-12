"""Raw and geometry-based feature extraction for GestureX.

Raw features preserve MediaPipe's landmark order as 63 values:
``landmark_0_x, landmark_0_y, landmark_0_z, ..., landmark_20_z``.

The eight invariant features centre points at the wrist (landmark 0), divide
five distances by the wrist-to-middle-MCP scale, and use three joint angles.
Consequently they are insensitive to global translation and uniform scale;
joint angles are additionally unaffected by in-plane rotation.  Degenerate
hands with no usable scale fail explicitly rather than producing unstable,
misleadingly huge values.
"""

from __future__ import annotations

from typing import Any, Final, Mapping, Sequence

import numpy as np

try:  # Supports direct scripts and package imports.
    from .config import (
        INVARIANT_FEATURE_COLUMNS,
        INVARIANT_FEATURE_DIMENSION,
        INVARIANT_REPRESENTATION,
        NORMALIZATION_EPSILON,
        RAW_FEATURE_COLUMNS,
        RAW_FEATURE_DIMENSION,
        RAW_REPRESENTATION,
    )
    from .landmarks import (
        LandmarkValidationError,
        flatten_landmarks,
        unflatten_landmarks,
        validate_landmarks,
    )
except ImportError:  # pragma: no cover - exercised by direct script execution.
    from config import (  # type: ignore
        INVARIANT_FEATURE_COLUMNS,
        INVARIANT_FEATURE_DIMENSION,
        INVARIANT_REPRESENTATION,
        NORMALIZATION_EPSILON,
        RAW_FEATURE_COLUMNS,
        RAW_FEATURE_DIMENSION,
        RAW_REPRESENTATION,
    )
    from landmarks import (  # type: ignore
        LandmarkValidationError,
        flatten_landmarks,
        unflatten_landmarks,
        validate_landmarks,
    )


class FeatureExtractionError(ValueError):
    """Raised when valid-looking landmarks cannot yield stable features."""


# MediaPipe hand landmark indices used by the 8-D feature vector.
WRIST: Final[int] = 0
THUMB_CMC: Final[int] = 1
THUMB_MCP: Final[int] = 2
THUMB_TIP: Final[int] = 4
INDEX_MCP: Final[int] = 5
INDEX_PIP: Final[int] = 6
INDEX_TIP: Final[int] = 8
MIDDLE_MCP: Final[int] = 9
MIDDLE_PIP: Final[int] = 10
MIDDLE_TIP: Final[int] = 12
RING_MCP: Final[int] = 13
RING_TIP: Final[int] = 16
PINKY_MCP: Final[int] = 17
PINKY_TIP: Final[int] = 20


def _coerce_landmarks(landmarks: Any) -> np.ndarray:
    """Accept either a 21x3 object or one flattened raw 63-D sample."""

    # Preserve MediaPipe landmark objects without trying ``np.asarray`` first.
    if hasattr(landmarks, "landmark"):
        return validate_landmarks(landmarks)
    try:
        tentative = np.asarray(landmarks)
    except (TypeError, ValueError):
        return validate_landmarks(landmarks)

    if tentative.ndim == 1 and tentative.size == RAW_FEATURE_DIMENSION:
        return unflatten_landmarks(tentative)
    return validate_landmarks(landmarks)


def normalize_representation(representation: str) -> str:
    """Resolve friendly representation spellings to ``raw`` or ``invariant``."""

    if not isinstance(representation, str):
        raise FeatureExtractionError("Feature representation must be a string.")
    normalized = representation.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "raw": RAW_REPRESENTATION,
        "raw_coordinates": RAW_REPRESENTATION,
        "raw_coordinate": RAW_REPRESENTATION,
        "invariant": INVARIANT_REPRESENTATION,
        "invariant_features": INVARIANT_REPRESENTATION,
        "geometric": INVARIANT_REPRESENTATION,
        "geometry": INVARIANT_REPRESENTATION,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise FeatureExtractionError(
            f"Unsupported feature representation {representation!r}; use 'raw' or 'invariant'."
        ) from exc


def feature_names(representation: str) -> tuple[str, ...]:
    """Return the ordered column names for a representation."""

    selected = normalize_representation(representation)
    if selected == RAW_REPRESENTATION:
        return RAW_FEATURE_COLUMNS
    return INVARIANT_FEATURE_COLUMNS


get_feature_names = feature_names


def extract_raw_features(landmarks: Any) -> np.ndarray:
    """Return exactly 63 raw MediaPipe coordinates in documented landmark order."""

    try:
        return flatten_landmarks(_coerce_landmarks(landmarks))
    except LandmarkValidationError as exc:
        raise FeatureExtractionError(str(exc)) from exc


def euclidean_distance(first: Sequence[float], second: Sequence[float]) -> float:
    """Calculate a three-dimensional Euclidean distance."""

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != (3,) or second_array.shape != (3,):
        raise FeatureExtractionError("Distances require two XYZ vectors of shape (3,).")
    return float(np.linalg.norm(first_array - second_array))


def angle_at_vertex(
    first: Sequence[float],
    vertex: Sequence[float],
    last: Sequence[float],
    *,
    epsilon: float = NORMALIZATION_EPSILON,
) -> float:
    """Return the interior angle ``first-vertex-last`` in degrees, from 0 to 180."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    first_array = np.asarray(first, dtype=np.float64)
    vertex_array = np.asarray(vertex, dtype=np.float64)
    last_array = np.asarray(last, dtype=np.float64)
    if any(array.shape != (3,) for array in (first_array, vertex_array, last_array)):
        raise FeatureExtractionError("Angles require three XYZ vectors of shape (3,).")

    left = first_array - vertex_array
    right = last_array - vertex_array
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= epsilon or right_norm <= epsilon:
        raise FeatureExtractionError(
            "Cannot calculate a joint angle from a zero-length landmark segment."
        )
    cosine = float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _normalization_scale(centered_landmarks: np.ndarray, epsilon: float) -> float:
    """Use wrist-to-middle-MCP distance as the per-hand scale reference."""

    scale = float(np.linalg.norm(centered_landmarks[MIDDLE_MCP]))
    if not np.isfinite(scale) or scale <= epsilon:
        raise FeatureExtractionError(
            "Cannot normalize invariant features: wrist-to-middle-MCP scale is "
            f"{scale:.3g}, which is at or below epsilon={epsilon:.3g}."
        )
    return scale


def extract_invariant_features(
    landmarks: Any,
    *,
    epsilon: float = NORMALIZATION_EPSILON,
) -> np.ndarray:
    """Extract the required 8-D translation- and scale-aware geometry vector.

    Feature order:

    1. thumb tip → index MCP distance / wrist → middle MCP distance
    2. index tip → index MCP distance / scale
    3. middle tip → middle MCP distance / scale
    4. ring tip → ring MCP distance / scale
    5. pinky tip → pinky MCP distance / scale
    6. thumb CMC–MCP–tip angle (degrees)
    7. index MCP–PIP–tip angle (degrees)
    8. middle MCP–PIP–tip angle (degrees)

    The wrist is subtracted from every landmark before all calculations.  This
    is mathematically redundant for pairwise distances but makes the reference
    frame explicit and prevents future coordinate features from drifting away
    from the same translation-normalized convention.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    try:
        points = _coerce_landmarks(landmarks)
    except LandmarkValidationError as exc:
        raise FeatureExtractionError(str(exc)) from exc

    centered = points - points[WRIST]
    scale = _normalization_scale(centered, epsilon)

    normalized_distances = (
        euclidean_distance(centered[THUMB_TIP], centered[INDEX_MCP]) / scale,
        euclidean_distance(centered[INDEX_TIP], centered[INDEX_MCP]) / scale,
        euclidean_distance(centered[MIDDLE_TIP], centered[MIDDLE_MCP]) / scale,
        euclidean_distance(centered[RING_TIP], centered[RING_MCP]) / scale,
        euclidean_distance(centered[PINKY_TIP], centered[PINKY_MCP]) / scale,
    )
    angles = (
        angle_at_vertex(centered[THUMB_CMC], centered[THUMB_MCP], centered[THUMB_TIP], epsilon=epsilon),
        angle_at_vertex(centered[INDEX_MCP], centered[INDEX_PIP], centered[INDEX_TIP], epsilon=epsilon),
        angle_at_vertex(centered[MIDDLE_MCP], centered[MIDDLE_PIP], centered[MIDDLE_TIP], epsilon=epsilon),
    )
    features = np.asarray((*normalized_distances, *angles), dtype=np.float64)
    if features.shape != (INVARIANT_FEATURE_DIMENSION,) or not np.isfinite(features).all():
        raise FeatureExtractionError("Invariant feature extraction produced invalid values.")
    return features


def extract_features(landmarks: Any, representation: str = RAW_REPRESENTATION) -> np.ndarray:
    """Extract either a raw 63-D or invariant 8-D feature vector."""

    selected = normalize_representation(representation)
    if selected == RAW_REPRESENTATION:
        return extract_raw_features(landmarks)
    return extract_invariant_features(landmarks)


def landmarks_from_row(
    row: Mapping[str, Any] | Sequence[float] | np.ndarray,
    *,
    columns: Sequence[str] = RAW_FEATURE_COLUMNS,
) -> np.ndarray:
    """Create a landmark array from a CSV/DataFrame row or a raw 63-D vector."""

    if isinstance(row, Mapping) or hasattr(row, "__getitem__") and not isinstance(
        row, (list, tuple, np.ndarray)
    ):
        try:
            values = [row[column] for column in columns]  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            raise FeatureExtractionError(
                "Row does not contain every required raw landmark feature column."
            ) from exc
    else:
        values = row
    try:
        return unflatten_landmarks(values)
    except LandmarkValidationError as exc:
        raise FeatureExtractionError(str(exc)) from exc


def feature_vector_as_dict(
    landmarks: Any,
    representation: str = RAW_REPRESENTATION,
) -> dict[str, float]:
    """Return an ordered-name dictionary convenient for CSV/DataFrame creation."""

    names = feature_names(representation)
    vector = extract_features(landmarks, representation)
    return dict(zip(names, map(float, vector), strict=True))


def transform_feature_frame(dataframe: Any, representation: str) -> Any:
    """Convert raw landmark columns in a DataFrame into an ordered feature DataFrame.

    Pandas is imported lazily so the numerical feature functions remain useful
    in minimal environments.  Invalid rows intentionally raise with their row
    index; silently dropping them would make class counts and session splits
    irreproducible.
    """

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise RuntimeError("pandas is required to transform a feature DataFrame.") from exc
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    selected = normalize_representation(representation)
    missing = [column for column in RAW_FEATURE_COLUMNS if column not in dataframe.columns]
    if missing:
        raise FeatureExtractionError(
            f"DataFrame is missing {len(missing)} raw landmark column(s), e.g. {missing[:3]}."
        )
    if selected == RAW_REPRESENTATION:
        values = dataframe.loc[:, list(RAW_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="raise")
        return values.copy()

    vectors: list[np.ndarray] = []
    for index, row in dataframe.loc[:, list(RAW_FEATURE_COLUMNS)].iterrows():
        try:
            vectors.append(extract_invariant_features(row.to_numpy()))
        except FeatureExtractionError as exc:
            raise FeatureExtractionError(
                f"Could not extract invariant features for row {index!r}: {exc}"
            ) from exc
    return pd.DataFrame(vectors, index=dataframe.index, columns=INVARIANT_FEATURE_COLUMNS)
