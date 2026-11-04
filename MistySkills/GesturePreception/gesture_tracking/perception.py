"""Gesture perception: find hands in the camera feed with MediaPipe Hand Landmarker.

Run the demo on its own with:  python -m gesture_tracking.perception
"""
import math
import sys
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
STOP_HOLD_SEC = 0.3        # how long the pose must be held before firing
STOP_MAX_TILT_DEG = 35     # how far from vertical the hand may lean
STOP_MAX_MOVE = 0.15       # how far the palm may drift while held, in hand sizes

# WAVE (open hand moving side to side)
WAVE_WINDOW_SEC = 2.0      # how much palm history to keep
WAVE_MIN_SWINGS = 3        # direction changes needed (3 = right-left-right-left; 2 let hand repositioning count)
WAVE_MIN_MOVE = 0.4        # smallest move that counts as a swing, in hand sizes
WAVE_MAX_GAP_SEC = 0.3     # if the hand stops looking open for longer than this, the wave starts over
WAVE_TURN_FRAMES = 2       # frames in a row showing the other side of the hand before it counts as turned around
GESTURE_COOLDOWN_SEC = 2.0 # quiet time after a WAVE, COME or GO_AWAY before the same gesture can fire again
AFTER_GESTURE_QUIET_SEC = 1.0  # after any gesture, WAVE/COME/GO_AWAY can't build up while the hand moves to its next pose (STOP still can)

# COME and GO_AWAY (hand bending at the wrist and back). Which way the fingers
FLICK_WINDOW_SEC = 1.5     # how much reach history to keep
FLICK_MIN_SWINGS = 2       # bend/straighten direction changes needed
FLICK_MIN_BEND = 0.5       # smallest change in reach that counts, in palm widths
FLICK_MAX_DRIFT = 1.0      # how far the palm may slide sideways meanwhile, in palm widths
FLICK_MIN_DIRECTION_SHARE = 0.6  # share of frames in the window where the fingers must point the right way
FLICK_MIN_PALM_WIDTH = 0.4 # knuckles narrower than this (in hand sizes) mean the hand is turning over: the flick starts over

# Landmark indices
WRIST = 0
INDEX_KNUCKLE = 5
MIDDLE_KNUCKLE = 9
MIDDLE_TIP = 12
PINKY_KNUCKLE = 17
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]  # (tip, middle joint): index, middle, ring, pinky
FINGERTIPS = [8, 12, 16, 20]                      # index, middle, ring, pinky
KNUCKLES = [5, 9, 13, 17]                         # the bases of those fingers
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
    """How long the hand looks (wrist to middle fingertip), in palm widths"""
    return math.dist(points[WRIST], points[MIDDLE_TIP]) / palm_width(points)


def is_open_hand(points):
    """True if the index, middle, ring and pinky fingers are all stretched out."""
    wrist = points[WRIST]
    return all(math.dist(wrist, points[tip]) > math.dist(wrist, points[joint])
               for tip, joint in FINGERS)


def fingers_point_down(points):
    """True if the fingertips are lower than the knuckles, on average (image y grows downward)."""
    tips_y = sum(points[i][1] for i in FINGERTIPS) / len(FINGERTIPS)
    knuckles_y = sum(points[i][1] for i in KNUCKLES) / len(KNUCKLES)
    return tips_y > knuckles_y


def palm_faces_camera(points, label):
    """True if the palm faces the camera, False if the back of the hand does, None if unknown"""
    if label not in ("Left", "Right"):
        return None
    wrist, index, pinky = points[WRIST], points[INDEX_KNUCKLE], points[PINKY_KNUCKLE]
    turn = ((index[0] - wrist[0]) * (pinky[1] - wrist[1]) - (index[1] - wrist[1]) * (pinky[0] - wrist[0]))
    if turn == 0:
        return None
    right_hand = label == "Right" 
    return turn < 0 if right_hand else turn > 0


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
    """Fires once when a raised open palm (palm should be facing the robot)"""

    def __init__(self):
        self.anchor = None
        self.start_time = None
        self.fired = False

    def update(self, points, now, palm_toward_camera=None):
        """Feed one frame's hand (or None) and which side faces the camera
        (see palm_faces_camera). Returns True on the frame STOP fires.
        """
        back_of_hand = palm_toward_camera is False
        if points is None or back_of_hand or not (is_open_hand(points) and is_raised(points)):
            self.anchor = None
            self.fired = False
            return False

        center = palm_center(points)
        if self.anchor is None or math.dist(center, self.anchor) > STOP_MAX_MOVE * hand_size(points):
            # Pose just started, or the palm moved too much -> start timing again
            self.anchor = center
            self.start_time = now
            return False

        if not self.fired and now - self.start_time >= STOP_HOLD_SEC:
            self.fired = True
            return True
        return False


def count_swings(xs, min_move):
    """Count how many times a list of values (palm x positions, finger reach) changes direction"""
    swings = 0
    direction = 0      # +1 moving right, -1 moving left, 0 not decided yet
    turn_x = xs[0]     # the farthest point reached in the current direction
    low = high = xs[0] 
    for x in xs[1:]:
        if direction == 0:
            low, high = min(low, x), max(high, x)
            if x - low >= min_move:
                direction, turn_x = 1, x
            elif high - x >= min_move:
                direction, turn_x = -1, x
        elif (x - turn_x) * direction > 0:
            turn_x = x  # still going the same way
        elif abs(x - turn_x) >= min_move:
            swings += 1  # direction changed -> count as a swing
            direction = -direction
            turn_x = x
    return swings


class WaveDetector:
    """Fires when an open hand swings side to side WAVE_MIN_SWINGS times,
    keeping the same side of the hand toward Misty"""

    def __init__(self):
        self.history = deque()     # (time, palm x) for the last WAVE_WINDOW_SEC
        self.cooldown_until = 0.0 
        self.side = None           
        self.other_side_frames = 0 

    def reset(self):
        """Forget the wave in progress."""
        self.history.clear()
        self.side = None
        self.other_side_frames = 0

    def update(self, points, now, palm_toward_camera=None):
        """Update the wave detector with a new frame of hand points and the current time.
        Returns True if a wave is detected."""
        if now < self.cooldown_until:
            return False
        if points is not None and is_open_hand(points):
            if self.history and now - self.history[-1][0] > WAVE_MAX_GAP_SEC:
                self.reset()  # the hand wasn't open for a while
            self.track_side(palm_toward_camera)
            self.history.append((now, palm_center(points)[0]))
        while self.history and now - self.history[0][0] > WAVE_WINDOW_SEC:
            self.history.popleft()  # forget positions older than the window

        if points is None or len(self.history) < 2:
            return False
        xs = [x for _, x in self.history]
        if count_swings(xs, WAVE_MIN_MOVE * hand_size(points)) >= WAVE_MIN_SWINGS:
            self.reset()
            self.cooldown_until = now + GESTURE_COOLDOWN_SEC
            return True
        return False

    def track_side(self, palm):
        """Start the wave over if the hand has turned around for WAVE_TURN_FRAMES frames."""
        if palm is None:
            return 
        if self.side is None or palm == self.side:
            self.side = palm
            self.other_side_frames = 0
            return
        self.other_side_frames += 1
        if self.other_side_frames >= WAVE_TURN_FRAMES:
            self.reset()
            self.side = palm


class FlickDetector:
    """Fires when the hand bends at the wrist and back FLICK_MIN_SWINGS times, with the fingers pointing the chosen way (up or down) for most of the window"""

    def __init__(self, fingers_down):
        self.fingers_down = fingers_down  # True for GO_AWAY, False for COME
        self.history = deque()     # (time, finger reach, palm x, fingers point down) for the last FLICK_WINDOW_SEC
        self.cooldown_until = 0.0  # no new gesture can fire before this time

    def reset(self):
        """Forget the flick in progress."""
        self.history.clear()

    def update(self, points, now):
        """Feed one frame's hand (or None). Returns True on the frame the gesture fires."""
        if now < self.cooldown_until:
            return False
        if points is not None and palm_width(points) <= FLICK_MIN_PALM_WIDTH * hand_size(points):
            self.reset()
            return False
        if points is not None:
            self.history.append((now, finger_reach(points), palm_center(points)[0], fingers_point_down(points)))
        while self.history and now - self.history[0][0] > FLICK_WINDOW_SEC:
            self.history.popleft()  # forget values older than the window

        if points is None or len(self.history) < 2:
            return False
        xs = [x for _, _, x, _ in self.history]
        if max(xs) - min(xs) > FLICK_MAX_DRIFT * palm_width(points):
            return False  # the palm slid sideways: that's a wave or a passing hand, not a flick
        right_way = sum(1 for *_, down in self.history if down == self.fingers_down)
        if right_way < FLICK_MIN_DIRECTION_SHARE * len(self.history):
            return False  # the fingers mostly point the other way: that's the other gesture
        reaches = [r for _, r, _, _ in self.history]
        if count_swings(reaches, FLICK_MIN_BEND) >= FLICK_MIN_SWINGS:
            self.history.clear()
            self.cooldown_until = now + GESTURE_COOLDOWN_SEC
            return True
        return False


class ComeDetector(FlickDetector):
    def __init__(self):
        super().__init__(fingers_down=False)


class GoAwayDetector(FlickDetector):
    def __init__(self):
        super().__init__(fingers_down=True)


class HandTracker:
    """Reads frames from a camera and finds the hands in each one"""
    def __init__(self, source=0, rotate=None):
        self.rotate = rotate
        if isinstance(source, int) and sys.platform == "win32":
            # Windows' default camera system -> only for testing without robot
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source {source!r}")

        self.latest = None
        self.new_frame = threading.Condition()
        self.running = True
        self.reader = threading.Thread(target=self.read_loop, daemon=True)
        self.reader.start()

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_timestamp_ms = -1

    def read_loop(self):
        """read frames as fast as they arrive, keep only the newest."""
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.new_frame:
                self.latest = frame
                self.new_frame.notify()

    def read(self):
        """Take the newest frame and detect hands.
        Returns (frame, hands, labels)"""
        with self.new_frame:
            self.new_frame.wait_for(lambda: self.latest is not None, timeout=1.0)
            frame, self.latest = self.latest, None
        if frame is None:
            return None, [], []
        if self.rotate is not None:
            frame = cv2.rotate(frame, self.rotate)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = max(int(time.monotonic() * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        labels = [h[0].category_name if h else None for h in result.handedness]
        return frame, result.hand_landmarks, labels

    def close(self):
        self.running = False
        self.reader.join(timeout=1.0)
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
        event = p.poll()   # "WAVE", "STOP", "COME", "GO_AWAY" or None, called once per loop
        key = p.last_key   # key pressed in the video window, e.g. "f", or None
        p.close()
    """

    def __init__(self, source=0, show=True, tracker=None, rotate=None):
        self.show = show
        self.last_key = None 
        self.last_event = None
        self.last_event_time = None
        self.tracker = tracker or HandTracker(source, rotate)
        self.stop_detector = StopDetector()
        self.wave_detector = WaveDetector()
        self.come_detector = ComeDetector()
        self.go_away_detector = GoAwayDetector()
        self.quiet_until = 0.0

    def poll(self):
        """Read one frame and return "WAVE", "STOP", "COME", "GO_AWAY" or None."""
        frame, hands, labels = self.tracker.read()
        self.last_key = None
        if frame is None:
            return None
        now = time.monotonic()
        height, width = frame.shape[:2]

        closest = pick_closest_hand(hands)
        points = to_points(closest, width, height) if closest else None
        label = next((l for h, l in zip(hands, labels) if h is closest), None)
        palm = palm_faces_camera(points, label) if points else None


        movers = (self.wave_detector, self.come_detector, self.go_away_detector)
        if now < self.quiet_until:
            for detector in movers:
                detector.reset()

        stop = self.stop_detector.update(points, now, palm)
        wave = self.wave_detector.update(points, now, palm)
        come = self.come_detector.update(points, now)
        go_away = self.go_away_detector.update(points, now)
        event = ("STOP" if stop else "WAVE" if wave else "COME" if come
                 else "GO_AWAY" if go_away else None)
        if event:
            self.last_event, self.last_event_time = event, now
            self.quiet_until = now + AFTER_GESTURE_QUIET_SEC
            for detector in movers:
                detector.reset()

        if self.show:
            self.draw(frame, hands, closest, points, palm, now)
            key = cv2.waitKey(1) & 0xFF
            if key != 0xFF:  # 0xFF means no key was pressed
                self.last_key = chr(key)
        return event

    def draw(self, frame, hands, closest, points, palm, now):
        """Show the frame with hand dots, status text and the last gesture."""
        for hand in hands:
            color = (0, 255, 0) if hand is closest else (160, 160, 160)
            draw_hand(frame, hand, color)

        status = f"hands: {len(hands)}"
        if points:
            status += f"  open: {is_open_hand(points)}  raised: {is_raised(points)}"
            status += "  palm: " + {True: "toward", False: "back", None: "?"}[palm]
            status += "  fingers: " + ("down" if fingers_point_down(points) else "up")
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
        self.tracker.close()
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
