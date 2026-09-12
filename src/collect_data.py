"""Webcam data collection for GestureX.

Opens the default (or specified) camera, runs MediaPipe on every frame, and
saves a CSV row only when a valid 21-landmark hand is detected.  Each row
carries five metadata fields (sample_id, session_id, subject_id, gesture,
timestamp) plus the 63 raw landmark coordinates in the documented order::

    landmark_0_x, landmark_0_y, landmark_0_z, ..., landmark_20_x, landmark_20_y, landmark_20_z

Usage (interactive mode)::

    python src/collect_data.py

Usage (fully automatic, e.g. for scripting)::

    python src/collect_data.py \\
        --gesture thumbs_up \\
        --session-id train_s01 \\
        --subject-id subject_01 \\
        --samples 200 \\
        --camera 0

The collector prevents saving invalid or no-hand samples, shows a progress
counter in the OpenCV window and on stdout, and appends to an existing session
CSV without touching other sessions.  Separate sessions use different output
files so cross-session and same-session splits remain clean.

Leakage note
------------
Collecting all gestures in one long, near-static burst and later splitting
the adjacent rows at random is a common leakage risk.  This script records
session and subject IDs for every sample so :mod:`src.train` can separate
sessions without relying on a purely random row split.  Collect an
*independent* cross-session with meaningfully different conditions (lighting,
distance, angle, background, or person).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Bootstrap: allow ``python src/collect_data.py`` from the repository root.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DEFAULT_CAMERA_INDEX,
    DEFAULT_SAMPLES_PER_GESTURE,
    GESTURE_DISPLAY_NAMES,
    GESTURE_LABELS,
    LANDMARK_COUNT,
    METADATA_COLUMNS,
    RAW_DATA_DIR,
    RAW_FEATURE_COLUMNS,
    canonicalize_gesture_label,
    ensure_project_directories,
)
from src.landmarks import (  # noqa: E402
    HandLandmarkDetector,
    draw_hand_annotations,
    flatten_landmarks,
)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
_CSV_FIELDNAMES: tuple[str, ...] = METADATA_COLUMNS + RAW_FEATURE_COLUMNS


def _session_csv_path(session_id: str) -> Path:
    """Each session writes to its own file to prevent accidental row interleaving."""

    safe_id = session_id.strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    return RAW_DATA_DIR / f"session_{safe_id}.csv"


def _load_existing_sample_count(path: Path) -> int:
    """Count rows already saved for this session without loading them into memory."""

    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for _ in reader:
            count += 1
    return count


def _next_sample_id(path: Path) -> int:
    """Return a strictly increasing integer sample index for this session file."""

    return _load_existing_sample_count(path)


# ---------------------------------------------------------------------------
# Interactive prompting helpers
# ---------------------------------------------------------------------------
def _prompt_gesture() -> str:
    """Show a numbered menu and return a canonical gesture label."""

    print("\nAvailable gesture classes:")
    for index, label in enumerate(GESTURE_LABELS, start=1):
        display = GESTURE_DISPLAY_NAMES.get(label, label)
        print(f"  {index:>2}. {display} ({label})")
    while True:
        raw = input("Enter gesture number or label: ").strip()
        # Accept a number.
        if raw.isdigit():
            choice = int(raw) - 1
            if 0 <= choice < len(GESTURE_LABELS):
                return GESTURE_LABELS[choice]
            print(f"  Please enter a number between 1 and {len(GESTURE_LABELS)}.")
            continue
        # Accept a label string.
        try:
            return canonicalize_gesture_label(raw)
        except ValueError as exc:
            print(f"  {exc}")


def _prompt_session_id() -> str:
    """Ask for a session identifier and enforce a non-empty string."""

    while True:
        value = input("Session ID (e.g. train_s01, test_s01): ").strip()
        if value:
            return value
        print("  Session ID cannot be empty.")


def _prompt_subject_id() -> str:
    """Ask for a pseudonymous subject identifier."""

    while True:
        value = input("Subject ID (e.g. subject_01): ").strip()
        if value:
            return value
        print("  Subject ID cannot be empty.")


def _prompt_sample_count(default: int) -> int:
    while True:
        raw = input(f"Samples per gesture [{default}]: ").strip()
        if not raw:
            return default
        if raw.isdigit() and int(raw) >= 1:
            return int(raw)
        print("  Enter a positive integer.")


# ---------------------------------------------------------------------------
# Frame drawing helpers
# ---------------------------------------------------------------------------
_PROGRESS_BAR_WIDTH = 260


def _draw_collection_overlay(
    frame: np.ndarray,
    gesture: str,
    collected: int,
    target: int,
    session_id: str,
    subject_id: str,
    paused: bool,
    hand_present: bool,
) -> None:
    """Draw a semi-transparent status panel over the webcam frame in-place."""

    height, width = frame.shape[:2]
    panel_h = 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_h), (15, 20, 30), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    display_name = GESTURE_DISPLAY_NAMES.get(gesture, gesture)
    status_colour = (50, 200, 50) if hand_present and not paused else (0, 180, 255)
    state_text = "Paused – press Space to resume" if paused else (
        "Collecting …" if hand_present else "No hand detected"
    )
    cv2.putText(frame, f"Gesture: {display_name}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, state_text, (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_colour, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Session: {session_id}  |  Subject: {subject_id}",
                (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)

    # Progress bar.
    fraction = min(collected / max(target, 1), 1.0)
    bar_x, bar_y, bar_h = 10, 96, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + _PROGRESS_BAR_WIDTH, bar_y + bar_h),
                  (60, 60, 60), -1)
    filled_width = int(_PROGRESS_BAR_WIDTH * fraction)
    if filled_width > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_h),
                      (50, 205, 50), -1)
    cv2.putText(frame, f"{collected}/{target}",
                (bar_x + _PROGRESS_BAR_WIDTH + 8, bar_y + bar_h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA)

    cv2.putText(frame, "Space=pause  Q/Esc=quit  N=next gesture",
                (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 210), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Core collection loop
# ---------------------------------------------------------------------------
def collect_gesture_session(
    *,
    gesture: str,
    session_id: str,
    subject_id: str,
    target_samples: int,
    camera_index: int,
    interval_ms: int,
    mirror: bool,
) -> int:
    """Collect landmarks for one gesture into the session CSV.

    Returns the number of new samples written.  The function appends rows to
    an existing session CSV (or creates it with the header) so multiple gestures
    can be added in the same session without overwriting earlier data.

    Leakage note: the function never reads or validates other sessions' files.
    The CSV per-session separation makes it impossible for the splitter to
    confuse which rows belong to the held-out cross-session.
    """

    ensure_project_directories()
    canonical_gesture = canonicalize_gesture_label(gesture)
    output_path = _session_csv_path(session_id)
    write_header = not output_path.exists()
    sample_id_offset = _next_sample_id(output_path)

    collected = 0
    last_save_time = 0.0

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Try --camera 1, check permissions, or close another webcam application."
        )

    paused = False
    print(f"\n→ Recording '{canonical_gesture}' | session={session_id} | subject={subject_id}")
    print(f"  Target: {target_samples} samples  |  Saving to: {output_path}")
    print("  Press SPACE to pause/resume, N to skip to next gesture, Q/Esc to quit.")

    try:
        with HandLandmarkDetector() as detector, \
             output_path.open("a", newline="", encoding="utf-8") as csv_file:

            writer = csv.DictWriter(csv_file, fieldnames=list(_CSV_FIELDNAMES))
            if write_header:
                writer.writeheader()

            while collected < target_samples:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Webcam returned an empty frame; check camera connection.")
                if mirror:
                    frame = cv2.flip(frame, 1)

                detection = detector.detect(frame)
                hand_present = detection is not None
                now = time.monotonic()
                saved_this_frame = False

                if not paused and hand_present and (now - last_save_time) >= (interval_ms / 1000.0):
                    coords = flatten_landmarks(detection.landmarks)  # type: ignore[union-attr]
                    row: dict[str, object] = {
                        "sample_id": f"{session_id}_{sample_id_offset + collected:06d}",
                        "session_id": session_id,
                        "subject_id": subject_id,
                        "gesture": canonical_gesture,
                        "timestamp": now,
                    }
                    for col_name, coord_value in zip(RAW_FEATURE_COLUMNS, coords):
                        row[col_name] = float(coord_value)
                    writer.writerow(row)
                    csv_file.flush()
                    collected += 1
                    last_save_time = now
                    saved_this_frame = True
                    # Progress every 10 samples.
                    if collected % 10 == 0 or collected == target_samples:
                        print(f"  {collected}/{target_samples}", end="\r", flush=True)

                if hand_present and detection is not None:
                    draw_hand_annotations(
                        frame,
                        detection.landmarks,
                        bounding_box=detection.bounding_box,
                    )

                _draw_collection_overlay(
                    frame,
                    canonical_gesture,
                    collected,
                    target_samples,
                    session_id,
                    subject_id,
                    paused,
                    hand_present,
                )
                cv2.imshow("GestureX — Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # Q or Esc → abort entirely.
                    print(f"\n  Collection aborted after {collected} samples.")
                    return collected
                if key == ord(" "):
                    paused = not paused
                    state = "paused" if paused else "resumed"
                    print(f"\n  Collection {state}. Press SPACE to toggle.")
                if key == ord("n"):
                    print(f"\n  Skipping to next gesture after {collected} samples.")
                    return collected

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n  ✓ Collected {collected} samples for '{canonical_gesture}'.")
    return collected


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GestureX webcam data collector. "
            "Saves MediaPipe landmarks to CSV with session metadata. "
            "Run without arguments for interactive mode."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gesture",
        choices=list(GESTURE_LABELS),
        default=None,
        help="Gesture class to collect. Prompted interactively when omitted.",
    )
    parser.add_argument(
        "--all-gestures",
        action="store_true",
        help="Collect every configured gesture class in sequence.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Recording session identifier, e.g. train_s01 or test_s01. Prompted if omitted.",
    )
    parser.add_argument(
        "--subject-id",
        default=None,
        help="Pseudonymous subject identifier, e.g. subject_01. Prompted if omitted.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES_PER_GESTURE,
        help="Target samples per gesture.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help="OpenCV camera index (0 for the default webcam).",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=100,
        help="Minimum milliseconds between saved samples to reduce near-duplicate frames.",
    )
    parser.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        help="Do not flip the preview horizontally.",
    )
    parser.set_defaults(mirror=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.samples < 1:
        parser.error("--samples must be a positive integer.")
    if args.interval_ms < 0:
        parser.error("--interval-ms cannot be negative.")

    # Resolve session and subject IDs (CLI or prompt).
    session_id: str = args.session_id or _prompt_session_id()
    subject_id: str = args.subject_id or _prompt_subject_id()

    # Resolve which gestures to collect.
    if args.all_gestures:
        gestures_to_collect: list[str] = list(GESTURE_LABELS)
    elif args.gesture:
        gestures_to_collect = [args.gesture]
    else:
        gestures_to_collect = [_prompt_gesture()]
        while True:
            again = input("Add another gesture? [y/N]: ").strip().lower()
            if again in ("y", "yes"):
                gestures_to_collect.append(_prompt_gesture())
            else:
                break

    print(
        f"\n{'='*60}\n"
        f"  GestureX collection plan\n"
        f"  Session:   {session_id}\n"
        f"  Subject:   {subject_id}\n"
        f"  Gestures:  {', '.join(gestures_to_collect)}\n"
        f"  Target:    {args.samples} samples per gesture\n"
        f"  Output:    {_session_csv_path(session_id)}\n"
        f"{'='*60}"
    )

    total_collected = 0
    for gesture in gestures_to_collect:
        try:
            count = collect_gesture_session(
                gesture=gesture,
                session_id=session_id,
                subject_id=subject_id,
                target_samples=args.samples,
                camera_index=args.camera,
                interval_ms=args.interval_ms,
                mirror=args.mirror,
            )
            total_collected += count
        except RuntimeError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            return 1

    print(f"\n{'='*60}")
    print(f"  Session '{session_id}' complete: {total_collected} total samples saved.")
    print(f"  CSV: {_session_csv_path(session_id)}")
    print(f"{'='*60}")
    print(
        "\nNext steps:\n"
        "  • Record additional sessions (different lighting / angle / person for cross-session).\n"
        "  • Merge CSVs and run:  python src/train.py --cross-session <held-out-session-id>\n"
        "  • Evaluate:            python src/evaluate.py\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
