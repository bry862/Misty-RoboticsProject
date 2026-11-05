#face_tracking.py
#
# Face recognition + head tracking math, shared by main.py and ../gesture_demo.py.

import os
import pickle
from pathlib import Path

import numpy as np
from insightface.app import FaceAnalysis

# ---------------------------------------------------------------------------
# Config (tune these on the real robot)
# ---------------------------------------------------------------------------
DB_FILE = Path(__file__).resolve().parent / "known_faces.pkl"  # next to this file, wherever you run from


MATCH_THRESHOLD = 0.45          # cosine similarity needed to call it a known face
ENROLL_BELOW = 0.30             # only enroll if the best score is clearly a stranger
MIN_ENROLL_DET_SCORE = 0.7      # don't enroll low-quality detections
MIN_ENROLL_FACE_PX = 90         # bbox width; skip tiny, far-away faces

TRACKING_DISTANCE_THRESHOLD_CM = 100.0
KNOWN_WIDTH_CM = 15.0
FOCAL_LENGTH_PIXELS_640 = 600.0  # calibrated for a 640px-wide frame; scaled below

# Head limits (keep in sync with misty_client.move_head_safely)
YAW_LIMIT = 40.0
PITCH_LIMIT = 25.0

# Degrees of head rotation per image pixel, same for both axes.
# Estimated from your log: ~157 px of shift for ~17 deg of yaw -> ~0.11 deg/px.
DEG_PER_PIXEL = 0.11

GAIN = 1.0                      # fraction of the measured error corrected per move (lower if it overshoots)

# Anti-jitter settings
START_MOVING_PX = 30           # centered -> must be off-center by this much to start moving
STOP_MOVING_PX = 20             # while moving, keep correcting until within this many px
MIN_MOVE_DEG = 1.0              # ignore corrections smaller than this
MAX_STEP_DEG = 25.0            # cap how far one update can move the head

SETTLE_SECONDS = 0.3           # ignore events after a move (picture capture adds ~0.9s on top)
HEAD_VELOCITY = 100             # NOTE: misty_client.move_head_safely clamps this; see its safe_vel line

# If the head moves the wrong way, flip these
YAW_SIGN = -1.0                 # face right of center -> negative yaw delta
PITCH_SIGN = 1.0                # face below center -> positive pitch delta


def clamp(value, low, high):
    return max(low, min(high, value))

def compute_cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def estimate_distance_cm(bbox, frame_width):
    face_pixel_width = bbox[2] - bbox[0]
    if face_pixel_width <= 0:
        return 0.0
    focal = FOCAL_LENGTH_PIXELS_640 * (frame_width / 640.0)
    return (KNOWN_WIDTH_CM * focal) / face_pixel_width

class FaceRecognizer:
    """
    Finds faces in a frame and remembers who they are (new people are saved
    as Guest_N in known_faces.pkl). Loading the model takes a few seconds.
    """

    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_s")
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

        if os.path.exists(DB_FILE):
            with open(DB_FILE, "rb") as f:
                self.known_faces = pickle.load(f)
            print(f"Loaded {len(self.known_faces)} known faces from disk.")
        else:
            self.known_faces = {}
            print("No existing face database found. Starting fresh.")

        self.last_recognized = None

    def save_database(self):
        with open(DB_FILE, "wb") as f:
            pickle.dump(self.known_faces, f)

    def identify_or_enroll(self, face, distance_cm):
        best_match, highest_score = None, 0.0
        for name, stored_vector in self.known_faces.items():
            score = compute_cosine_similarity(face.embedding, stored_vector)
            if score > highest_score:
                highest_score, best_match = score, name

        if highest_score > MATCH_THRESHOLD:
            # Only log on identity change; per-event spam was hiding the tracking logs
            if best_match != self.last_recognized:
                print(f"Recognized: {best_match} (Score: {highest_score:.2f}) | Distance: {distance_cm:.1f} cm")
                self.last_recognized = best_match
        elif (highest_score < ENROLL_BELOW
          and face.det_score >= MIN_ENROLL_DET_SCORE
          and (face.bbox[2] - face.bbox[0]) >= MIN_ENROLL_FACE_PX):
            new_id = f"Guest_{len(self.known_faces) + 1}"
            self.known_faces[new_id] = face.embedding
            self.save_database()
            print(f"New face stored as {new_id} | Distance: {distance_cm:.1f} cm")
        else:
            print(f"Low-quality detection (det_score {face.det_score:.2f}), not enrolling.")

    def find_closest(self, frame):
        """
        Identify every face in the frame (for logging / enrollment) and return
        (all faces, closest face, its distance in cm). Closest = largest bounding box width.
        Returns ([], None, 0.0) if there's no face.
        """
        faces = self.app.get(frame)
        if not faces:
            return [], None, 0.0

        frame_w = frame.shape[1]
        for face in faces:
            self.identify_or_enroll(face, estimate_distance_cm(face.bbox, frame_w))

        target = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])
        return faces, target, estimate_distance_cm(target.bbox, frame_w)

class HeadController:
    """
    Pure tracking math: no robot calls, no sleeping, no printing.
    Give it a face bounding box and frame size; it returns the new absolute
    (yaw, pitch) to command, or None if the head should stay put.
    All tuning of the control behavior happens in here.
    """

    def __init__(self):
        self.yaw = 0.0
        self.pitch = 0.0
        self.centered = True          # hysteresis state
        self.last_error = (0.0, 0.0)  # (error_x, error_y) in px, for debugging

    def reset(self, yaw=0.0, pitch=0.0):
        self.yaw, self.pitch, self.centered = yaw, pitch, True

    def compute(self, bbox, frame_w, frame_h):
        error_x = (bbox[0] + bbox[2]) / 2 - frame_w / 2
        error_y = (bbox[1] + bbox[3]) / 2 - frame_h / 2
        self.last_error = (error_x, error_y)
        off_center = max(abs(error_x), abs(error_y))

        # Hysteresis: hard to start moving, easy to keep going until centered.
        # This stops the head from re-correcting over small bbox jitter.
        if self.centered:
            if off_center < START_MOVING_PX:
                return None
            self.centered = False
        elif off_center < STOP_MOVING_PX:
            self.centered = True
            return None

        # Convert pixel error to degrees of head rotation
        yaw_delta = YAW_SIGN * error_x * DEG_PER_PIXEL * GAIN
        pitch_delta = PITCH_SIGN * error_y * DEG_PER_PIXEL * GAIN

        # Cap the step size
        yaw_delta = clamp(yaw_delta, -MAX_STEP_DEG, MAX_STEP_DEG)
        pitch_delta = clamp(pitch_delta, -MAX_STEP_DEG, MAX_STEP_DEG)

        # Clamp the *state* so it can't wind up beyond the real limits
        new_yaw = clamp(self.yaw + yaw_delta, -YAW_LIMIT, YAW_LIMIT)
        new_pitch = clamp(self.pitch + pitch_delta, -PITCH_LIMIT, PITCH_LIMIT)

        # Ignore tiny corrections
        if abs(new_yaw - self.yaw) < MIN_MOVE_DEG and abs(new_pitch - self.pitch) < MIN_MOVE_DEG:
            return None

        self.yaw, self.pitch = new_yaw, new_pitch
        return new_yaw, new_pitch
