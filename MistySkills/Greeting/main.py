#main.py

import time
import base64
import os
import pickle
import threading
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from misty_client import MistyClient

# ---------------------------------------------------------------------------
# Config (tune these on the real robot)
# ---------------------------------------------------------------------------
DB_FILE = "known_faces.pkl"


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

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=-1, det_size=(640, 640))

if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        known_faces = pickle.load(f)
    print(f"Loaded {len(known_faces)} known faces from disk.")
else:
    known_faces = {}
    print("No existing face database found. Starting fresh.")

client = None

# Tracking state
ignore_until = 0.0
processing_lock = threading.Lock()

# Logging / debug state
START_TIME = time.time()
last_recognized = {"name": None}
debug_state = {"saved_frame": False, "printed_event": False}

def clamp(value, low, high):
    return max(low, min(high, value))

def compute_cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def save_database():
    with open(DB_FILE, "wb") as f:
        pickle.dump(known_faces, f)

def grab_frame():
    """Takes a picture from Misty and returns an OpenCV frame (or None)."""
    response = client.take_picture_base64()

    img_base64 = None
    if isinstance(response, dict):
        result_node = response.get("result")
        if isinstance(result_node, dict):
            img_base64 = result_node.get("base64") or result_node.get("Image")
        if not img_base64:
            img_base64 = response.get("base64") or response.get("Image")
    elif isinstance(response, str):
        img_base64 = response

    if not img_base64:
        print(f"Could not parse image data. Raw response was: {response}")
        return None

    img_bytes = base64.b64decode(img_base64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        print("Failed to decode image frame into OpenCV format.")
    return frame

def estimate_distance_cm(bbox, frame_width):
    face_pixel_width = bbox[2] - bbox[0]
    if face_pixel_width <= 0:
        return 0.0
    focal = FOCAL_LENGTH_PIXELS_640 * (frame_width / 640.0)
    return (KNOWN_WIDTH_CM * focal) / face_pixel_width

def identify_or_enroll(face, distance_cm):
    best_match, highest_score = None, 0.0
    for name, stored_vector in known_faces.items():
        score = compute_cosine_similarity(face.embedding, stored_vector)
        if score > highest_score:
            highest_score, best_match = score, name

    if highest_score > MATCH_THRESHOLD:
        # Only log on identity change; per-event spam was hiding the tracking logs
        if best_match != last_recognized["name"]:
            print(f"Recognized: {best_match} (Score: {highest_score:.2f}) | Distance: {distance_cm:.1f} cm")
            last_recognized["name"] = best_match
    elif (highest_score < ENROLL_BELOW
      and face.det_score >= MIN_ENROLL_DET_SCORE
      and (face.bbox[2] - face.bbox[0]) >= MIN_ENROLL_FACE_PX):
        new_id = f"Guest_{len(known_faces) + 1}"
        known_faces[new_id] = face.embedding
        save_database()
        print(f"New face stored as {new_id} | Distance: {distance_cm:.1f} cm")
    else:
        print(f"Low-quality detection (det_score {face.det_score:.2f}), not enrolling.")

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

head_controller = HeadController()

def track_face(bbox, frame_w, frame_h, note=""):
    """Thin wrapper: ask the controller what to do, then talk to the robot."""
    global ignore_until

    command = head_controller.compute(bbox, frame_w, frame_h)
    if command is None:
        return

    yaw, pitch = command
    ex, ey = head_controller.last_error
    t = time.time() - START_TIME
    print(f"[{t:7.1f}s] Error ({ex:+.0f}, {ey:+.0f}) px | Head -> Yaw {yaw:.1f}° | Pitch {pitch:.1f}° | {note}")
    client.move_head_safely(pitch=pitch, yaw=yaw, velocity=HEAD_VELOCITY)

    # Ignore events until the head has (mostly) stopped moving
    ignore_until = time.time() + SETTLE_SECONDS

def handle_face_event(data):
    if not debug_state["printed_event"]:
        print(f"[debug] raw event data: {data}")
        debug_state["printed_event"] = True

    # Drop events while the head is settling or another event is processing
    if time.time() < ignore_until:
        return
    if not processing_lock.acquire(blocking=False):
        return

    try:
        t0 = time.time()
        frame = grab_frame()
        t_capture = time.time() - t0
        if frame is None:
            return

        t1 = time.time()
        faces = app.get(frame)
        t_infer = time.time() - t1
        if not faces:
            print("InsightFace couldn't detect a face in the snapshot frame.")
            return

        frame_h, frame_w = frame.shape[:2]

        # Identify every face for logging / enrollment
        for face in faces:
            identify_or_enroll(face, estimate_distance_cm(face.bbox, frame_w))

        # Track ONLY the closest face (largest bounding box width)
        target = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])
        distance_cm = estimate_distance_cm(target.bbox, frame_w)

        if not debug_state["saved_frame"]:
            dbg = frame.copy()
            x1, y1, x2, y2 = [int(v) for v in target.bbox]
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.drawMarker(dbg, (frame_w // 2, frame_h // 2), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.imwrite("debug_frame.jpg", dbg)
            print(f"[debug] frame shape (h, w, c) = {frame.shape}; saved debug_frame.jpg")
            debug_state["saved_frame"] = True

        if 0 < distance_cm <= TRACKING_DISTANCE_THRESHOLD_CM:
            track_face(target.bbox, frame_w, frame_h,
                       note=f"capture {t_capture:.2f}s, infer {t_infer:.2f}s")
        else:
            print(f"-> Closest face ({distance_cm:.1f} cm) outside tracking threshold.")

    except Exception as e:
        print(f"Error during face processing pipeline: {e}")
    finally:
        processing_lock.release()

if __name__ == "__main__":
    MISTY_IP = "192.168.1.5"

    print(f"Connecting to Misty at {MISTY_IP}...")
    client = MistyClient(MISTY_IP)

    # Center the head first so our yaw/pitch estimate matches reality
    client.move_head_safely(pitch=0, yaw=0, velocity=HEAD_VELOCITY)
    time.sleep(1.5)
    head_controller.reset(yaw=0.0, pitch=0.0)

    print("Starting face recognition... Look at Misty's camera.")
    client.start_face_recognition(handle_face_event)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down and stopping face recognition...")
        client.stop_face_recognition()

# # ## To do's: 
# 1. Re center head when no one is detected
# 2. Aside from only registering faces that are accurate enough, 
# we need to only register if it within the threshold. 

# 3. Running main positions Misty's head up slightly, 
# sincne it will be on the floor most of the time.  