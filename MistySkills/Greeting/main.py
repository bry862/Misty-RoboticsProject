#main.py

import argparse
import time
import base64
import threading
import cv2
import numpy as np
from face_tracking import (FaceRecognizer, HeadController, TRACKING_DISTANCE_THRESHOLD_CM, SETTLE_SECONDS, HEAD_VELOCITY)
from misty_client import MistyClient

# Tuning settings live in face_tracking.py (shared with ../gesture_demo.py)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
recognizer = FaceRecognizer()

client = None

# Tracking state
ignore_until = 0.0
processing_lock = threading.Lock()

# Logging / debug state
START_TIME = time.time()
debug_state = {"saved_frame": False, "printed_event": False}

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

        # Identifies every face (logging / enrollment); track ONLY the closest one
        t1 = time.time()
        faces, target, distance_cm = recognizer.find_closest(frame)
        t_infer = time.time() - t1
        if not faces:
            print("InsightFace couldn't detect a face in the snapshot frame.")
            return

        frame_h, frame_w = frame.shape[:2]

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
    parser = argparse.ArgumentParser(description="Misty recognizes faces and turns her head toward the closest one.")
    parser.add_argument("--ip", required=True, help="Misty's IP address")
    MISTY_IP = parser.parse_args().ip

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