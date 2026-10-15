"""Checkpoint 1 demo: Misty reacts to hand gestures.

    STOP -> Misty stops driving (also cuts a COME drive short)
    WAVE -> Misty greets (green LED, raises an arm, says hi)
    COME -> Misty drives slowly forward for 5 seconds

Uses Misty's official Python SDK (mistyPy). Run from the project folder:
    python gesture_demo.py --ip 192.168.1.5             # use Misty's own camera
    python gesture_demo.py --ip 192.168.1.5 --webcam    # backup: use this computer's webcam
    python gesture_demo.py --webcam --no-robot          # test gestures without Misty

Press q in the video window to quit.
"""
import argparse
import os
import threading
import time

import cv2
import mistyPy.RobotCommands
import requests
from mistyPy.Robot import Robot

from gesture_tracking.perception import GesturePerception

# --- Reaction settings (check these on the robot) ---
VIDEO_PORT = 1936           # port Misty streams her camera on (RTSP)
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
CAMERA_ROTATION = cv2.ROTATE_90_CLOCKWISE  # Misty's camera streams sideways; use ROTATE_90_COUNTERCLOCKWISE if upside down
COME_SPEED = 20             # forward speed for COME, -100 to 100 (keep it slow)
COME_TIME_MS = 5000         # how long Misty drives forward for COME (a STOP gesture cuts it short)
ARM_UP, ARM_DOWN = -80, 90  # arm positions in degrees for the greeting wave
REQUEST_TIMEOUT_SEC = 3     # give up on a command if Misty doesn't answer in time


def _request_with_timeout(verb, url, **kwargs):
    kwargs.setdefault("timeout", REQUEST_TIMEOUT_SEC)
    return requests.request(verb, url, **kwargs)


# The SDK sends commands with no timeout, so an unreachable Misty would freeze
# gesture detection (including STOP) for ~20 s per command. Give it a limit.
mistyPy.RobotCommands.request = _request_with_timeout

# Ask OpenCV's video decoder not to buffer Misty's stream, to cut delay.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "fflags;nobuffer|flags;low_delay")


def check(response):
    """Warn if Misty answered a command with an error (the SDK doesn't raise)."""
    if not response.ok:
        print(f"  Misty returned an error: {response.status_code} {response.text[:100]}")


def greet(misty):
    """WAVE: green LED, raise the right arm and say hi; the arm lowers after 2 s."""
    check(misty.change_led(red=0, green=255, blue=0))
    check(misty.move_arms(rightArmPosition=ARM_UP, rightArmVelocity=80))
    check(misty.speak(text="Hi! I'm Misty! How are you doing today?"))
    threading.Timer(2.0, lambda: misty.move_arms(rightArmPosition=ARM_DOWN, rightArmVelocity=50)).start()


def come_forward(misty):
    """COME: blue LED, drive slowly forward for COME_TIME_MS; Misty stops by herself unless STOP cuts it short."""
    check(misty.change_led(red=0, green=0, blue=255))
    check(misty.drive_time(linearVelocity=COME_SPEED, angularVelocity=0, timeMs=COME_TIME_MS))


def react(misty, event):
    """Send Misty the reaction for one gesture."""
    if event == "STOP":
        check(misty.stop())
    elif event == "WAVE":
        greet(misty)
    elif event == "COME":
        come_forward(misty)


def start_video_stream(misty):
    """Start streaming Misty's camera and return the address to read it from.

    Turning on AV streaming switches off Misty's normal camera service
    (the two can't run at the same time).
    """
    check(misty.enable_av_streaming_service())
    check(misty.start_av_streaming(url=f"rtspd:{VIDEO_PORT}", width=VIDEO_WIDTH, height=VIDEO_HEIGHT))
    return f"rtsp://{misty.ip}:{VIDEO_PORT}"


def stop_video_stream(misty):
    """Stop streaming and give Misty her normal camera back."""
    misty.stop_av_streaming()
    misty.disable_av_streaming_service()
    misty.enable_camera_service()


def open_perception(source, rotate=None, attempts=5):
    """Open the video source, retrying while Misty's stream starts up."""
    for attempt in range(1, attempts + 1):
        try:
            return GesturePerception(source=source, show=True, rotate=rotate)
        except RuntimeError:
            if attempt == attempts:
                raise
            print(f"Waiting for video stream... ({attempt}/{attempts})")
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description="Misty reacts to STOP, WAVE and COME gestures.")
    parser.add_argument("--ip", help="Misty's IP address")
    parser.add_argument("--webcam", action="store_true", help="use this computer's webcam, not Misty's camera")
    parser.add_argument("--no-robot", action="store_true", help="only print gestures, don't send commands")
    args = parser.parse_args()
    if not args.no_robot and not args.ip:
        parser.error("--ip is required unless you use --no-robot")
    if args.no_robot and not args.webcam:
        parser.error("--no-robot needs --webcam (there's no Misty camera without the robot)")

    misty = None if args.no_robot else Robot(args.ip)
    if args.webcam:
        source = 0
    else:
        print("Starting Misty's camera stream...")
        source = start_video_stream(misty)

    perception = open_perception(source, rotate=None if args.webcam else CAMERA_ROTATION)
    print("Ready. Show a gesture: STOP, WAVE or COME. Press q in the video window to quit.")
    try:
        while perception.last_key != "q":
            event = perception.poll()
            if not event:
                continue
            print(event)
            if misty:
                try:
                    react(misty, event)
                except requests.RequestException as error:
                    print(f"  Misty didn't respond: {error}")  # keep going, don't crash the demo
    finally:
        perception.close()
        if misty:
            try:
                misty.stop()  # never leave Misty driving
                if not args.webcam:
                    stop_video_stream(misty)
            except requests.RequestException:
                pass


if __name__ == "__main__":
    main()
