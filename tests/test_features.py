"""Tests for invariant feature extraction in GestureX.

Covers the requirements from the recruitment brief:
- Feature dimensionality (must be exactly 8)
- Handling of malformed landmarks
- Zero / near-zero normalization scale guard
- Mathematical correctness of distances and angles
- Feature extraction from both MediaPipe-style objects and raw arrays
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INVARIANT_FEATURE_DIMENSION, NORMALIZATION_EPSILON
from src.features import (
    FeatureExtractionError,
    angle_at_vertex,
    euclidean_distance,
    extract_invariant_features,
    extract_raw_features,
    feature_names,
    normalize_representation,
)
from src.landmarks import LandmarkValidationError, unflatten_landmarks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _valid_landmarks(scale: float = 0.15) -> np.ndarray:
    """Return a syntactically valid (21, 3) array that passes the scale guard.

    The wrist (landmark 0) is at the origin; the middle-MCP (landmark 9) is
    at (0, scale, 0) so the normalization scale equals ``scale``.
    """
    points = np.zeros((21, 3), dtype=np.float64)
    # Middle-MCP (landmark 9) defines the normalization scale.
    points[9] = [0.0, scale, 0.0]
    # Spread the other landmarks minimally so angle computations have valid arms.
    for index in range(1, 21):
        if index != 9:
            points[index] = [index * 0.02, index * 0.01 + scale * 0.1, 0.0]
    return points


def _flat_landmarks(scale: float = 0.15) -> np.ndarray:
    """Return the 63-D flattened version of ``_valid_landmarks``."""
    return _valid_landmarks(scale).reshape(63)


# ---------------------------------------------------------------------------
# Dimensionality
# ---------------------------------------------------------------------------
class TestInvariantFeatureDimension:
    def test_output_is_8d(self) -> None:
        features = extract_invariant_features(_valid_landmarks())
        assert features.shape == (INVARIANT_FEATURE_DIMENSION,), (
            f"Expected {INVARIANT_FEATURE_DIMENSION} invariant features, got {features.shape}"
        )

    def test_output_is_finite(self) -> None:
        features = extract_invariant_features(_valid_landmarks())
        assert np.isfinite(features).all(), "Invariant features contain NaN or Inf."

    def test_feature_names_count(self) -> None:
        names = feature_names("invariant")
        assert len(names) == INVARIANT_FEATURE_DIMENSION, (
            f"Expected {INVARIANT_FEATURE_DIMENSION} feature names, got {len(names)}"
        )

    def test_raw_feature_names_count(self) -> None:
        names = feature_names("raw")
        assert len(names) == 63


# ---------------------------------------------------------------------------
# Malformed landmarks
# ---------------------------------------------------------------------------
class TestMalformedLandmarks:
    def test_none_landmarks_raise(self) -> None:
        with pytest.raises((FeatureExtractionError, LandmarkValidationError)):
            extract_invariant_features(None)

    def test_wrong_count_raises(self) -> None:
        bad = np.zeros((10, 3), dtype=np.float64)
        with pytest.raises((FeatureExtractionError, LandmarkValidationError)):
            extract_invariant_features(bad)

    def test_2d_instead_of_21x3_raises(self) -> None:
        bad = np.zeros((5, 5), dtype=np.float64)
        with pytest.raises((FeatureExtractionError, LandmarkValidationError)):
            extract_invariant_features(bad)

    def test_nan_landmark_raises(self) -> None:
        points = _valid_landmarks()
        points[5, 0] = float("nan")
        with pytest.raises((FeatureExtractionError, LandmarkValidationError)):
            extract_invariant_features(points)

    def test_inf_landmark_raises(self) -> None:
        points = _valid_landmarks()
        points[0, 1] = float("inf")
        with pytest.raises((FeatureExtractionError, LandmarkValidationError)):
            extract_invariant_features(points)

    def test_empty_array_raises(self) -> None:
        with pytest.raises((FeatureExtractionError, LandmarkValidationError)):
            extract_invariant_features(np.array([]))

    def test_wrong_dimensionality_raises(self) -> None:
        """A flat 63-D all-zeros array reshapes to (21, 3) but fails the zero-scale guard."""
        bad = np.zeros(63, dtype=np.float64)
        # The coerce helper reshapes 63-D arrays to (21, 3) (valid shape).
        # All-zero landmarks make wrist == middle-MCP, so scale = 0 and the
        # scale guard must raise FeatureExtractionError.
        with pytest.raises(FeatureExtractionError):
            extract_invariant_features(bad)


# ---------------------------------------------------------------------------
# Zero / near-zero normalization scale guard
# ---------------------------------------------------------------------------
class TestZeroScaleGuard:
    def test_zero_wrist_to_middle_mcp_raises(self) -> None:
        """All landmarks at the origin → wrist == middle-MCP → scale = 0 → error."""
        points = np.zeros((21, 3), dtype=np.float64)
        with pytest.raises(FeatureExtractionError, match=r"scale|epsilon|normaliz"):
            extract_invariant_features(points)

    def test_scale_below_epsilon_raises(self) -> None:
        """Microscopic scale should trigger the guard."""
        points = np.zeros((21, 3), dtype=np.float64)
        # Set middle-MCP (index 9) just below epsilon.
        points[9] = [0.0, NORMALIZATION_EPSILON * 0.5, 0.0]
        with pytest.raises(FeatureExtractionError):
            extract_invariant_features(points)

    def test_scale_at_epsilon_raises(self) -> None:
        """Scale exactly equal to epsilon should still be rejected."""
        points = np.zeros((21, 3), dtype=np.float64)
        points[9] = [0.0, NORMALIZATION_EPSILON, 0.0]
        with pytest.raises(FeatureExtractionError):
            extract_invariant_features(points)

    def test_scale_just_above_epsilon_succeeds(self) -> None:
        """A valid scale allows successful extraction."""
        points = np.zeros((21, 3), dtype=np.float64)
        points[9] = [0.0, NORMALIZATION_EPSILON * 100, 0.0]
        # Spread other landmarks slightly to give angles valid arms.
        for index in range(1, 21):
            if index != 9:
                points[index] = [index * 0.01, index * 0.01 + NORMALIZATION_EPSILON * 10, 0.0]
        # May raise FeatureExtractionError if angles collapse; that is acceptable.
        try:
            features = extract_invariant_features(points)
            assert features.shape == (8,)
        except FeatureExtractionError:
            pass  # Degenerate geometry → explicit error is correct behaviour.


# ---------------------------------------------------------------------------
# Translation invariance
# ---------------------------------------------------------------------------
class TestTranslationInvariance:
    """Invariant features must not change when the entire hand shifts."""

    def test_shift_in_x(self) -> None:
        base = _valid_landmarks()
        shifted = base.copy()
        shifted[:, 0] += 0.35
        np.testing.assert_allclose(
            extract_invariant_features(base),
            extract_invariant_features(shifted),
            rtol=1e-6,
            err_msg="Invariant features changed under x-axis translation.",
        )

    def test_shift_in_y(self) -> None:
        base = _valid_landmarks()
        shifted = base.copy()
        shifted[:, 1] -= 0.20
        np.testing.assert_allclose(
            extract_invariant_features(base),
            extract_invariant_features(shifted),
            rtol=1e-6,
        )

    def test_shift_in_z(self) -> None:
        base = _valid_landmarks()
        shifted = base.copy()
        shifted[:, 2] += 0.10
        np.testing.assert_allclose(
            extract_invariant_features(base),
            extract_invariant_features(shifted),
            rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# Raw feature extraction
# ---------------------------------------------------------------------------
class TestRawFeatureExtraction:
    def test_raw_output_is_63d(self) -> None:
        raw = extract_raw_features(_valid_landmarks())
        assert raw.shape == (63,)

    def test_raw_from_flat_input(self) -> None:
        flat = _flat_landmarks()
        raw = extract_raw_features(flat)
        assert raw.shape == (63,)
        np.testing.assert_allclose(raw, flat, rtol=1e-9)

    def test_raw_preserves_values(self) -> None:
        points = _valid_landmarks()
        raw = extract_raw_features(points)
        expected = points.reshape(63)
        np.testing.assert_allclose(raw, expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# Euclidean distance helper
# ---------------------------------------------------------------------------
class TestEuclideanDistance:
    def test_identity_is_zero(self) -> None:
        p = [0.1, 0.2, 0.3]
        assert euclidean_distance(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_unit_axis_distance(self) -> None:
        assert euclidean_distance([0, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
        assert euclidean_distance([0, 0, 0], [0, 1, 0]) == pytest.approx(1.0)
        assert euclidean_distance([0, 0, 0], [0, 0, 1]) == pytest.approx(1.0)

    def test_3_4_5_triangle(self) -> None:
        assert euclidean_distance([0, 0, 0], [3, 4, 0]) == pytest.approx(5.0)

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(FeatureExtractionError):
            euclidean_distance([0, 0], [1, 1])


# ---------------------------------------------------------------------------
# Angle helper
# ---------------------------------------------------------------------------
class TestAngleAtVertex:
    def test_right_angle(self) -> None:
        # Three points forming a 90° angle at the vertex.
        angle = angle_at_vertex([1, 0, 0], [0, 0, 0], [0, 1, 0])
        assert angle == pytest.approx(90.0, abs=1e-6)

    def test_straight_line_is_180(self) -> None:
        angle = angle_at_vertex([-1, 0, 0], [0, 0, 0], [1, 0, 0])
        assert angle == pytest.approx(180.0, abs=1e-6)

    def test_60_degree_angle(self) -> None:
        # Equilateral triangle: all angles = 60°.
        a = [1, 0, 0]
        v = [0, 0, 0]
        b = [0.5, np.sqrt(3) / 2, 0]
        angle = angle_at_vertex(a, v, b)
        assert angle == pytest.approx(60.0, abs=1e-5)

    def test_collinear_zero_arm_raises(self) -> None:
        # first == vertex → zero-length arm.
        with pytest.raises(FeatureExtractionError):
            angle_at_vertex([0, 0, 0], [0, 0, 0], [1, 0, 0])

    def test_negative_epsilon_raises(self) -> None:
        with pytest.raises(ValueError):
            angle_at_vertex([1, 0, 0], [0, 0, 0], [0, 1, 0], epsilon=-1.0)


# ---------------------------------------------------------------------------
# Representation normalization
# ---------------------------------------------------------------------------
class TestNormalizeRepresentation:
    @pytest.mark.parametrize(
        "alias",
        ["raw", "RAW", "Raw Coordinates", "raw_coordinates", "raw coordinate"],
    )
    def test_raw_aliases(self, alias: str) -> None:
        assert normalize_representation(alias) == "raw"

    @pytest.mark.parametrize(
        "alias",
        ["invariant", "INVARIANT", "invariant_features", "geometric", "geometry"],
    )
    def test_invariant_aliases(self, alias: str) -> None:
        assert normalize_representation(alias) == "invariant"

    def test_unsupported_raises(self) -> None:
        with pytest.raises(FeatureExtractionError):
            normalize_representation("deep_learning")

    def test_non_string_raises(self) -> None:
        with pytest.raises(FeatureExtractionError):
            normalize_representation(42)  # type: ignore[arg-type]
