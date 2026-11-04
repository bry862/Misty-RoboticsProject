"""Checkpoint 1 demo: Misty reacts to hand gestures.

    STOP    -> Misty stops driving and freezes her head (also cuts a COME or GO_AWAY short)
    WAVE    -> Misty greets (green LED, raises an arm, says hi)
    COME    -> Misty drives slowly forward for 5 seconds, slowly lifting her head to look up at you
    GO_AWAY -> Misty backs up for 3 seconds, slowly lowering her head like a shy pet

Uses Misty's official Python SDK (mistyPy). Run from the project folder:
    python gesture_demo.py --ip 192.168.1.5
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

# --- Reaction settings ---> we can change these ones
VIDEO_PORT = 1936
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
CAMERA_ROTATION = cv2.ROTATE_90_CLOCKWISE
COME_SPEED = 20
COME_TIME_MS = 5000
GO_AWAY_SPEED = 15
GO_AWAY_TIME_MS = 3000
ARM_UP, ARM_DOWN = -80, 90

HEAD_LEVEL = 0              # looking straight ahead
HEAD_UP = -25               # looking up at the face of someone standing close
HEAD_DOWN = 15              # looking down, shy, while backing away
HEAD_RESET_SEC = 2.0        # how long the head takes to come back level after GO_AWAY
REQUEST_TIMEOUT_SEC = 3     # give up on a command if Misty doesn't answer in time


def request_with_timeout(verb, url, **kwargs):
    kwargs.setdefault("timeout", REQUEST_TIMEOUT_SEC)
    return requests.request(verb, url, **kwargs)


# The SDK sends commands with no timeout, so an unreachable Misty would freeze
# gesture detection (including STOP) for ~20 s per command. Give it a limit.
mistyPy.RobotCommands.request = request_with_timeout

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


def move_head(misty, pitch, seconds):
    """Tilt Misty's head to pitch (degrees), taking the given number of seconds."""
    check(misty.move_head(pitch=pitch, roll=0, yaw=0, duration=seconds, units="degrees"))


head_reset = None  # pending timer that brings the head back level after GO_AWAY


def cancel_head_reset():
    """Forget a pending head reset, so it can't move the head during a newer reaction."""
    global head_reset
    if head_reset:
        head_reset.cancel()
        head_reset = None


def come_forward(misty):
    check(misty.change_led(red=0, green=0, blue=255))
    move_head(misty, HEAD_UP, COME_TIME_MS / 1000)
    check(misty.drive_time(linearVelocity=COME_SPEED, angularVelocity=0, timeMs=COME_TIME_MS))


def go_away(misty):
    global head_reset
    check(misty.change_led(red=255, green=80, blue=0))
    move_head(misty, HEAD_DOWN, GO_AWAY_TIME_MS / 1000)
    check(misty.drive_time(linearVelocity=-GO_AWAY_SPEED, angularVelocity=0, timeMs=GO_AWAY_TIME_MS))
    head_reset = threading.Timer(GO_AWAY_TIME_MS / 1000 + 0.5,
                                  lambda: move_head(misty, HEAD_LEVEL, HEAD_RESET_SEC))
    head_reset.start()


def react(misty, event):
    """Send Misty the reaction for one gesture."""
    if event in ("STOP", "COME", "GO_AWAY"):
        cancel_head_reset()
    if event == "STOP":
        check(misty.stop())
        check(misty.halt())  # also stops the head mid-tilt
    elif event == "WAVE":
        greet(misty)
    elif event == "COME":
        come_forward(misty)
    elif event == "GO_AWAY":
        go_away(misty)


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
    parser = argparse.ArgumentParser(description="Misty reacts to STOP, WAVE, COME and GO_AWAY gestures.")
    parser.add_argument("--ip", help="Misty's IP address")
    parser.add_argument("--webcam", action="store_true", help="use this computer's webcam, not Misty's camera")
    parser.add_argument("--no-robot", action="store_true", help="only print gestures, don't send commands")
    args = parser.parse_args()
    if not args.no_robot and not args.ip:
        parser.error("--ip is required unless you use --no-robot")
    if args.no_robot and not args.webcam:
        parser.error("--no-robot needs --webcam (there's no Misty camera without the robot)")

    misty = None if args.no_robot else Robot(args.ip)
    if misty:
        move_head(misty, HEAD_LEVEL, 1.0)  # so COME's head lift starts from level
    if args.webcam:
        source = 0
    else:
        print("Starting Misty's camera stream...")
        source = start_video_stream(misty)

    perception = open_perception(source, rotate=None if args.webcam else CAMERA_ROTATION)
    print("Ready. Show a gesture: STOP, WAVE, COME or GO_AWAY. Press q in the video window to quit.")
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
        cancel_head_reset()
        if misty:
            try:
                misty.stop()  # never leave Misty driving
                if not args.webcam:
                    stop_video_stream(misty)
            except requests.RequestException:
                pass


if __name__ == "__main__":
    main()
