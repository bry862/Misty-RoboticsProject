"""Gesture perception: find hands in the camera feed with MediaPipe Hand Landmarker.

Run the demo on its own with:  python -m gesture_tracking.perception
"""
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision


MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.6

# STOP (raised open palm)
STOP_HOLD_SEC = 0.4        # how long the pose must be held before firing
STOP_MAX_TILT_DEG = 35     # how far from vertical the hand may lean
STOP_MAX_MOVE = 0.15       # how far the palm may drift while held, in hand sizes

# Landmark indices
WRIST = 0
MIDDLE_KNUCKLE = 9
MIDDLE_TIP = 12
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]  # (tip, middle joint): index, middle, ring, pinky
PALM = [0, 5, 9, 13, 17]                          # wrist and the four finger bases


def hand_box_area(hand):
    """Area of the box around a hand's 21 landmarks, as a fraction of the frame"""
    xs = [lm.x for lm in hand]
    ys = [lm.y for lm in hand]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def pick_closest_hand(hands):
    """Return the closest to the camera hand, or None"""
    if not hands:
        return None
    return max(hands, key=hand_box_area)


def to_points(hand, width, height):
    """Turn a hand's landmarks into a list of 21 (x, y) pixel positions."""
    return [(lm.x * width, lm.y * height) for lm in hand]


def hand_size(points):
    """Distance from the wrist to the middle knuckle, used as a ruler."""
    return math.dist(points[WRIST], points[MIDDLE_KNUCKLE])


def palm_center(points):
    """Average position of the wrist and the four finger bases."""
    xs = [points[i][0] for i in PALM]
    ys = [points[i][1] for i in PALM]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def is_open_hand(points):
    """True if the index, middle, ring and pinky fingers are all stretched out."""
    wrist = points[WRIST]
    return all(math.dist(wrist, points[tip]) > math.dist(wrist, points[joint])
               for tip, joint in FINGERS)


def is_raised(points):
    """True if the fingers point up and the hand is close to vertical."""
    wrist = points[WRIST]
    knuckle = points[MIDDLE_KNUCKLE]
    if points[MIDDLE_TIP][1] >= wrist[1]:  # y grows downwards, so the tip must have a smaller y
        return False
    up = wrist[1] - knuckle[1]
    if up <= 0:
        return False
    sideways = abs(knuckle[0] - wrist[0])
    return math.degrees(math.atan2(sideways, up)) <= STOP_MAX_TILT_DEG


class StopDetector:
    """Fires once when a raised open palm is held still for STOP_HOLD_SEC."""

    def __init__(self):
        self._anchor = None      # palm position when the still pose began
        self._start_time = None  # when the still pose began
        self._fired = False      # True until the hand drops, so STOP fires only once

    def update(self, points, now):
        """Feed one frame's hand (or None). Returns True on the frame STOP fires."""
        if points is None or not (is_open_hand(points) and is_raised(points)):
            self._anchor = None
            self._fired = False
            return False

        center = palm_center(points)
        if self._anchor is None or math.dist(center, self._anchor) > STOP_MAX_MOVE * hand_size(points):
            # Pose just started, or the palm moved too much: start timing again from here.
            self._anchor = center
            self._start_time = now
            return False

        if not self._fired and now - self._start_time >= STOP_HOLD_SEC:
            self._fired = True
            return True
        return False


class HandTracker:
    """Reads frames from a camera and finds the hands in each one"""
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source {source!r}")

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def read(self):
        """Grab one frame and detect hands.

        Returns (frame, hands): frame is the BGR image (None if the camera gave
        nothing), and hands is a list of hands, each a list of 21 landmarks.
        """
        ok, frame = self.cap.read()
        if not ok:
            return None, []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return frame, result.hand_landmarks

    def close(self):
        self.landmarker.close()
        self.cap.release()


def draw_hand(frame, hand, color):
    """Draw a dot on each of the hand's 21 landmarks."""
    height, width = frame.shape[:2]
    for lm in hand:
        cv2.circle(frame, (int(lm.x * width), int(lm.y * height)), 4, color, -1)


def main():
    tracker = HandTracker(source=0)
    stop_detector = StopDetector()
    last_stop_time = -10.0
    try:
        while True:
            frame, hands = tracker.read()
            if frame is None:
                break
            now = time.monotonic()
            height, width = frame.shape[:2]

            closest = pick_closest_hand(hands)
            for hand in hands:
                color = (0, 255, 0) if hand is closest else (160, 160, 160)
                draw_hand(frame, hand, color)

            points = to_points(closest, width, height) if closest else None
            if stop_detector.update(points, now):
                last_stop_time = now
                print("STOP")

            status = f"hands: {len(hands)}"
            if points:
                status += f"  open: {is_open_hand(points)}  raised: {is_raised(points)}"
            cv2.putText(frame, status + "  (q to quit)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if now - last_stop_time < 1.0:  # keep the label up for a second after firing
                cv2.putText(frame, "STOP", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.imshow("Gesture perception", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
