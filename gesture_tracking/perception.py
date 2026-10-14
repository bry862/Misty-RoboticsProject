"""Gesture perception: find hands in the camera feed with MediaPipe Hand Landmarker.

Run the demo on its own with:  python -m gesture_tracking.perception
"""
import math
import threading
import time
from collections import deque
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

# WAVE (open hand moving side to side)
WAVE_WINDOW_SEC = 1.5      # how much palm history to keep
WAVE_MIN_SWINGS = 2        # direction changes needed to count as a wave
WAVE_MIN_MOVE = 0.3        # smallest move that counts as a swing, in hand sizes
GESTURE_COOLDOWN_SEC = 2.0 # quiet time after a WAVE or COME before the same gesture can fire again

# COME (hand bending toward the person and back, from the wrist)
COME_WINDOW_SEC = 1.5      # how much reach history to keep
COME_MIN_SWINGS = 2        # bend/straighten direction changes needed
COME_MIN_BEND = 0.5        # smallest change in reach that counts, in palm widths
COME_MAX_DRIFT = 1.0       # how far the palm may slide sideways meanwhile, in palm widths

# Landmark indices
WRIST = 0
INDEX_KNUCKLE = 5
MIDDLE_KNUCKLE = 9
MIDDLE_TIP = 12
PINKY_KNUCKLE = 17
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


def palm_width(points):
    """Distance across the knuckles, from the index finger base to the pinky base."""
    return math.dist(points[INDEX_KNUCKLE], points[PINKY_KNUCKLE])


def finger_reach(points):
    """How long the hand looks (wrist to middle fingertip), in palm widths.

    Bending the hand toward the person tips the fingers away from the camera,
    so the hand looks shorter and this number drops.
    """
    return math.dist(points[WRIST], points[MIDDLE_TIP]) / palm_width(points)


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


def count_swings(xs, min_move):
    """Count how many times a list of values (palm x positions, finger reach) changes direction.

    A move only counts once it has gone at least min_move from the last turning
    point, so small jitter is ignored.
    """
    swings = 0
    direction = 0      # +1 moving right, -1 moving left, 0 not decided yet
    turn_x = xs[0]     # the farthest point reached in the current direction
    for x in xs[1:]:
        if direction == 0:
            if abs(x - turn_x) >= min_move:
                direction = 1 if x > turn_x else -1
                turn_x = x
        elif (x - turn_x) * direction > 0:
            turn_x = x  # still going the same way: push the turning point further
        elif abs(x - turn_x) >= min_move:
            swings += 1  # came back far enough: that's a direction change
            direction = -direction
            turn_x = x
    return swings


class WaveDetector:
    """Fires when an open hand swings side to side WAVE_MIN_SWINGS times."""

    def __init__(self):
        self._history = deque()     # (time, palm x) for the last WAVE_WINDOW_SEC
        self._cooldown_until = 0.0  # no new wave can fire before this time

    def update(self, points, now):
        """Feed one frame's hand (or None). Returns True on the frame WAVE fires."""
        if now < self._cooldown_until:
            return False
        if points is not None and is_open_hand(points):
            self._history.append((now, palm_center(points)[0]))
        while self._history and now - self._history[0][0] > WAVE_WINDOW_SEC:
            self._history.popleft()  # forget positions older than the window

        if points is None or len(self._history) < 2:
            return False
        xs = [x for _, x in self._history]
        if count_swings(xs, WAVE_MIN_MOVE * hand_size(points)) >= WAVE_MIN_SWINGS:
            self._history.clear()
            self._cooldown_until = now + GESTURE_COOLDOWN_SEC
            return True
        return False


class ComeDetector:
    """Fires when the hand bends toward the person and back (a beckon) COME_MIN_SWINGS times."""

    def __init__(self):
        self._history = deque()     # (time, finger reach, palm x) for the last COME_WINDOW_SEC
        self._cooldown_until = 0.0  # no new COME can fire before this time

    def update(self, points, now):
        """Feed one frame's hand (or None). Returns True on the frame COME fires."""
        if now < self._cooldown_until:
            return False
        if points is not None and palm_width(points) > 0:
            self._history.append((now, finger_reach(points), palm_center(points)[0]))
        while self._history and now - self._history[0][0] > COME_WINDOW_SEC:
            self._history.popleft()  # forget values older than the window

        if points is None or len(self._history) < 2:
            return False
        xs = [x for _, _, x in self._history]
        if max(xs) - min(xs) > COME_MAX_DRIFT * palm_width(points):
            return False  # the palm slid sideways: that's a wave or a passing hand, not a beckon
        reaches = [r for _, r, _ in self._history]
        if count_swings(reaches, COME_MIN_BEND) >= COME_MIN_SWINGS:
            self._history.clear()
            self._cooldown_until = now + GESTURE_COOLDOWN_SEC
            return True
        return False


class HandTracker:
    """Reads frames from a camera and finds the hands in each one.

    A background thread keeps reading the camera and holds on to only the newest
    frame. Without it, frames queue up whenever detection (or a robot command)
    is slower than the camera, and we end up reacting to video from seconds ago.
    """
    def __init__(self, source=0, rotate=None):
        # rotate: a cv2.ROTATE_* code for cameras mounted sideways (like Misty's), or None
        self.rotate = rotate
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source {source!r}")

        self._latest = None                       # newest frame not yet handed out
        self._new_frame = threading.Condition()
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def _read_loop(self):
        """Background thread: read frames as fast as they arrive, keep only the newest."""
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)  # stream hiccup; try again
                continue
            with self._new_frame:
                self._latest = frame  # replaces any frame nobody picked up (that's the point)
                self._new_frame.notify()

    def read(self):
        """Take the newest frame and detect hands.

        Returns (frame, hands): frame is the BGR image (None if the camera gave
        nothing for a second), and hands is a list of hands, each a list of 21 landmarks.
        """
        with self._new_frame:
            self._new_frame.wait_for(lambda: self._latest is not None, timeout=1.0)
            frame, self._latest = self._latest, None
        if frame is None:
            return None, []
        if self.rotate is not None:
            frame = cv2.rotate(frame, self.rotate)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return frame, result.hand_landmarks

    def close(self):
        self._running = False
        self._reader.join(timeout=1.0)
        self.landmarker.close()
        self.cap.release()


def draw_hand(frame, hand, color):
    """Draw a dot on each of the hand's 21 landmarks."""
    height, width = frame.shape[:2]
    for lm in hand:
        cv2.circle(frame, (int(lm.x * width), int(lm.y * height)), 4, color, -1)


class GesturePerception:
    """The interface main.py uses.

        p = GesturePerception(source=0, show=True)
        event = p.poll()   # "WAVE", "STOP", "COME" or None, called once per loop
        key = p.last_key   # key pressed in the video window, e.g. "f", or None
        p.close()
    """

    def __init__(self, source=0, show=True, tracker=None, rotate=None):
        # tracker lets tests pass in a fake camera; normally leave it as None.
        self.show = show
        self.last_key = None          # key pressed in the video window during the last poll()
        self.last_event = None        # most recent gesture fired
        self.last_event_time = None   # time.monotonic() when it fired (for latency tests)
        self._tracker = tracker or HandTracker(source, rotate)
        self._stop_detector = StopDetector()
        self._wave_detector = WaveDetector()
        self._come_detector = ComeDetector()

    def poll(self):
        """Read one frame and return "WAVE", "STOP", "COME" or None."""
        frame, hands = self._tracker.read()
        self.last_key = None
        if frame is None:
            return None
        now = time.monotonic()
        height, width = frame.shape[:2]

        closest = pick_closest_hand(hands)
        points = to_points(closest, width, height) if closest else None

        # Run every detector each frame so each keeps its history up to date.
        # STOP wins if several fire at once, since it is the safety gesture.
        stop = self._stop_detector.update(points, now)
        wave = self._wave_detector.update(points, now)
        come = self._come_detector.update(points, now)
        event = "STOP" if stop else "WAVE" if wave else "COME" if come else None
        if event:
            self.last_event, self.last_event_time = event, now

        if self.show:
            self._draw(frame, hands, closest, points, now)
            key = cv2.waitKey(1) & 0xFF
            if key != 0xFF:  # 0xFF means no key was pressed
                self.last_key = chr(key)
        return event

    def _draw(self, frame, hands, closest, points, now):
        """Show the frame with hand dots, status text and the last gesture."""
        for hand in hands:
            color = (0, 255, 0) if hand is closest else (160, 160, 160)
            draw_hand(frame, hand, color)

        status = f"hands: {len(hands)}"
        if points:
            status += f"  open: {is_open_hand(points)}  raised: {is_raised(points)}"
            if palm_width(points) > 0:
                status += f"  reach: {finger_reach(points):.1f}"
        cv2.putText(frame, status + "  (q to quit)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if self.last_event and now - self.last_event_time < 1.0:  # keep the label up for a second
            cv2.putText(frame, self.last_event, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow("Gesture perception", frame)

    def close(self):
        """Release the camera and close the video window."""
        self._tracker.close()
        if self.show:
            cv2.destroyAllWindows()


def main():
    perception = GesturePerception(source=0, show=True)
    try:
        while perception.last_key != "q":
            event = perception.poll()
            if event:
                print(event)
    finally:
        perception.close()


if __name__ == "__main__":
    main()
