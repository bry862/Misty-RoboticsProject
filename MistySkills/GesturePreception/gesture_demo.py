"""Checkpoint 2 demo: 
Misty reacts to hand gestures and to being petted.

Hand gestures (seen by the camera):
    STOP    -> Misty stops driving and freezes her head (also cuts a COME or GO_AWAY short)
    WAVE    -> Misty greets (green LED, raises an arm, says hi)
    COME    -> Misty drives slowly forward for 5 seconds, slowly lifting her head to look up at you
    GO_AWAY -> Misty backs up for 3 seconds, slowly lowering her head like a shy pet

Pet interactions (Misty's touch sensors, see ../PetInteractions):
    Head pat -> happy/playful reaction (says something, plays an action like "hug" or "love")
    Chin pat -> curious/playful reaction ("surprise", "admire", ...)
    Every 3rd head pat -> excited: "walk-happy" dance, then back to neutral
    Misty ignores touches for 3 s after reacting.

Uses Misty's official Python SDK (mistyPy). Run from the project folder:
    python gesture_demo.py --ip 192.168.1.5
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import mistyPy.RobotCommands
import requests
from mistyPy.Events import Events
from mistyPy.Robot import Robot

from gesture_tracking.perception import GesturePerception

# Reuse the pet interaction logic from ../PetInteractions
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "PetInteractions"))
from pet_personality import PetPersonality 
from touch_listener import TouchListener 

# --- Reaction settings ---> we can change these ones
VIDEO_PORT = 1936
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
CAMERA_ROTATION = cv2.ROTATE_90_CLOCKWISE
COME_SPEED = 20
COME_TIME_MS = 5000
GO_AWAY_SPEED = 15
GO_AWAY_TIME_MS = 5000
ARM_UP, ARM_DOWN = -80, 90

HEAD_LEVEL = 0 
HEAD_UP = -25 
HEAD_DOWN = 15
HEAD_RESET_SEC = 2.0
REQUEST_TIMEOUT_SEC = 3


# --- Pet reaction settings ---
PET_LED_COLORS = {          # LED color per emotion, (red, green, blue)
    "happy": (255, 105, 180),
    "playful": (255, 140, 0),
    "curious": (255, 255, 0),
    "excited": (160, 32, 240),
}
EXCITED_DANCE_SEC = 2.0     # how long "walk-happy" plays before Misty resets to neutral


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


class PetReactions:
    """Pet interactions: picks a reaction with PetPersonality and plays it on Misty.

    Has a handle_touch(location) method so TouchListener can drive it directly.
    """

    def __init__(self, misty):
        self.misty = misty                # None = only print reactions
        self.personality = PetPersonality()

    def handle_touch(self, location):
        reaction = self.personality.react_to_touch(location)
        if reaction is None:
            return  # still cooling down from the last reaction
        print(f"PET {location}: {reaction['emotion']} -> \"{reaction['message']}\" ({reaction['action']})")
        if self.misty:
            try:
                self.play(reaction)
            except requests.RequestException as error:
                print(f"  Misty didn't respond: {error}")

    def play(self, reaction):
        check(self.misty.change_led(*PET_LED_COLORS.get(reaction["emotion"], (255, 255, 255))))
        check(self.misty.speak(text=reaction["message"]))
        check(self.misty.start_action(name=reaction["action"]))
        if reaction["emotion"] == "excited":
            # After the happy dance, put back to neutral pose
            threading.Timer(EXCITED_DANCE_SEC, lambda: self.misty.start_action(name="body-reset")).start()


def start_touch_sensors(misty, pet):
    """Listen to Misty's touch sensors and send head/chin pats to the pet reactions."""
    listener = TouchListener(pet)

    def on_touch(data):
        event = data.get("message", {})
        if isinstance(event, dict):
            listener.handle_sensor(event.get("sensorPosition", ""), event.get("isContacted", False))

    misty.register_event(event_type=Events.TouchSensor, event_name="PetTouch", debounce=200, keep_alive=True, callback_function=on_touch)


def start_video_stream(misty):
    """Start streaming Misty's camera and return the address to read it from"""
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

    pet = PetReactions(misty)
    if misty:
        start_touch_sensors(misty, pet)

    perception = open_perception(source, rotate=None if args.webcam else CAMERA_ROTATION)
    print("Ready. Show a gesture (STOP, WAVE, COME, GO_AWAY) or pet Misty's head/chin.")
    print("Keys in the video window: h = head pat, c = chin pat, q = quit.")
    try:
        while perception.last_key != "q":
            event = perception.poll()
            if perception.last_key == "h":
                pet.handle_touch("head")
            elif perception.last_key == "c":
                pet.handle_touch("chin")
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
                misty.unregister_event("PetTouch")
                misty.stop()  # never leave Misty driving
                if not args.webcam:
                    stop_video_stream(misty)
            except requests.RequestException:
                pass


if __name__ == "__main__":
    main()
