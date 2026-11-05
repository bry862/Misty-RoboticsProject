"""Checkpoint 2 demo: 
Misty reacts to hand gestures and to being petted.

Hand gestures (seen by the camera):
    STOP    -> Misty stops driving and freezes her head (also cuts a COME short, but not a GO_AWAY)
    WAVE    -> Misty greets (green LED, raises an arm, says hi)
    COME    -> Misty drives toward your face, turning to keep it ahead of her, and stops
               50 cm away (or, if she can't see a face, drives forward for 5 seconds,
               slowly lifting her head to look up at you)
    GO_AWAY -> Misty backs up for 1 second, says goodbye and turns around (180 degrees)

Face tracking (see ./Greeting/face_tracking.py):
    Misty turns her head toward the closest face within 1 m, and remembers new faces.
    During COME she follows your face at any distance. It pauses after STOP and during
    GO_AWAY. Needs insightface; turn off with --no-faces.

Pet interactions (Misty's touch sensors, see ./PetInteractions):
    Head pat -> happy/playful reaction (says something, plays an action like "hug" or "love")
    Chin pat -> curious/playful reaction ("surprise", "admire", ...)
    Every 3rd head pat -> excited: "walk-happy" dance, then back to neutral
    Misty ignores touches for 3 s after reacting.

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

# Reuse the gesture code from ./GesturePreception, the pet logic from ./PetInteractions and the face tracking from ./Greeting
SKILLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SKILLS_DIR / "GesturePreception"))
sys.path.insert(0, str(SKILLS_DIR / "PetInteractions"))
sys.path.insert(0, str(SKILLS_DIR / "Greeting"))
from gesture_tracking.perception import GesturePerception
from pet_personality import PetPersonality
from touch_listener import TouchListener

# --- Reaction settings ---> we can change these ones
VIDEO_PORT = 1936
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
CAMERA_ROTATION = cv2.ROTATE_90_CLOCKWISE
COME_SPEED = 20
COME_TIME_MS = 5000
GO_AWAY_SPEED = 15
GO_AWAY_BACKUP_MS = 1000
TURN_DEGREES = 180
TURN_SPEED = 40          # angular velocity, % of Misty's top turning speed
TURN_TIME_MS = 3000      # how long the turn may take
GOODBYE_SEC = 2.5        # how long saying GOODBYE takes; the turn starts after it
GO_AWAY_TIME_MS = GO_AWAY_BACKUP_MS + GOODBYE_SEC * 1000 + TURN_TIME_MS
ARM_UP, ARM_DOWN = -80, 90

HEAD_LEVEL = 0 
HEAD_UP = -25 
STOP_FREEZE_SEC = 3.0   # after STOP, face tracking leaves the head still this long
FACE_CHECK_SEC = 0.2    # pause between face checks, so gesture detection keeps enough CPU
FACE_SEEN_SEC = 1.5     # COME follows the face if it was seen this recently, else uses the fixed drive and head lift

# --- Following the person during COME ---
SAFE_DISTANCE_CM = 50   # Misty stops this far from the person's face
SLOW_DOWN_CM = 50       # she starts slowing down this much before the safe distance
FOLLOW_MIN_SPEED = 5    # slowest she drives while still approaching
FOLLOW_MAX_SEC = 10     # she gives up following after this long
FOLLOW_DRIVE_MS = 1500  # each drive command lasts this long, so she stops by herself if updates stop coming
FOLLOW_TURN_GAIN = 1.0  # turning speed (% of max) per degree the face is off to the side
FOLLOW_MAX_TURN = 30    # fastest she turns toward the face (% of max)
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


turn_timer = None  # pending timer for GO_AWAY's next step (goodbye, then turn)


def cancel_turn():
    """Forget GO_AWAY's pending steps, so they can't start during a newer reaction (or after STOP)."""
    global turn_timer
    if turn_timer:
        turn_timer.cancel()
        turn_timer = None


def come_forward(misty):
    """COME when Misty can't see a face: drive forward for a fixed time, slowly lifting her head."""
    check(misty.change_led(red=0, green=0, blue=255))
    move_head(misty, HEAD_UP, COME_TIME_MS / 1000)
    check(misty.drive_time(linearVelocity=COME_SPEED, angularVelocity=0, timeMs=COME_TIME_MS))


def go_away(misty):
    """Back up for a second, then say goodbye, then turn around. Each step waits for the one before."""
    global turn_timer
    check(misty.change_led(red=255, green=80, blue=0))
    check(misty.drive_time(linearVelocity=-GO_AWAY_SPEED, angularVelocity=0, timeMs=GO_AWAY_BACKUP_MS))
    turn_timer = threading.Timer(GO_AWAY_BACKUP_MS / 1000, lambda: say_bye(misty))
    turn_timer.start()


def say_bye(misty):
    global turn_timer
    try:
        check(misty.speak(text="Bye! Have a wonderful day!"))
    except requests.RequestException as error:
        print(f"  Misty didn't respond: {error}")
    turn_timer = threading.Timer(GOODBYE_SEC, lambda: turn_around(misty))
    turn_timer.start()


def turn_around(misty):
    try:
        check(misty.drive_time(linearVelocity=0, angularVelocity=TURN_SPEED, timeMs=TURN_TIME_MS, degree=TURN_DEGREES))
    except requests.RequestException as error:
        print(f"  Misty didn't respond: {error}")


def load_face_tracking():
    """Import Greeting/face_tracking.py, or return None if it can't load"""
    try:
        import face_tracking
    except ImportError as error:
        print(f"Face tracking is off, it couldn't load: {error}")
        return None
    return face_tracking


class FaceFollower:
    """Turns Misty's head toward the closest face (Greeting/face_tracking.py), and
    during COME drives her toward that face, stopping SAFE_DISTANCE_CM away.

    Uses the same camera frames as gesture detection, because Misty can't stream
    video and take pictures at the same time
    """

    def __init__(self, misty, perception, face_tracking):
        self.misty = misty
        self.perception = perception
        self.face_tracking = face_tracking
        self.recognizer = face_tracking.FaceRecognizer()
        self.head = face_tracking.HeadController()
        self.lock = threading.Lock()   # held while sending drive commands, so STOP always comes after them
        self.paused_until = 0.0
        self.following_until = 0.0     # during COME: drive toward the face until then
        self.follow_finished = False   # set when following ends by itself (reached the person, lost them, too long)
        self.last_face_time = 0.0      # when a face was last seen, at any distance
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def pause(self, seconds, yaw=None, pitch=None):
        """Leave the head to a gesture reaction for a while (also ends following).

        yaw/pitch: where the reaction leaves the head, so tracking continues from there.
        """
        with self.lock:
            self.paused_until = time.monotonic() + seconds
            self.following_until = 0.0
            if pitch is not None:
                self.head.reset(yaw=yaw, pitch=pitch)

    def sees_face(self):
        """True if a face was seen in the last FACE_SEEN_SEC seconds."""
        return time.monotonic() - self.last_face_time < FACE_SEEN_SEC

    def following(self):
        return time.monotonic() < self.following_until

    def start_following(self):
        """COME: drive toward the face and stop SAFE_DISTANCE_CM away from it.
        The body turns toward the face, so the head looks straight ahead (yaw 0) and only tilts."""
        with self.lock:
            self.paused_until = 0.0
            self.follow_finished = False
            self.following_until = time.monotonic() + FOLLOW_MAX_SEC
            self.head.reset(yaw=0, pitch=self.head.pitch)
            check(self.misty.move_head(pitch=self.head.pitch, roll=0, yaw=0,
                                       velocity=self.face_tracking.HEAD_VELOCITY, units="degrees"))

    def stop_following(self, reason):
        """End COME and stop driving. Call with self.lock held."""
        self.following_until = 0.0
        self.follow_finished = True
        print(f"COME finished: {reason}")
        check(self.misty.stop())

    def drive_toward(self, target, distance_cm, frame_w):
        """One step of following: steer toward the face, slow down near it. Call with self.lock held."""
        if distance_cm <= SAFE_DISTANCE_CM:
            self.stop_following(f"reached you ({distance_cm:.0f} cm away)")
            return
        error_x = (target.bbox[0] + target.bbox[2]) / 2 - frame_w / 2
        side_deg = self.face_tracking.YAW_SIGN * error_x * self.face_tracking.DEG_PER_PIXEL  # + = face is to her left
        turn = max(-FOLLOW_MAX_TURN, min(FOLLOW_MAX_TURN, FOLLOW_TURN_GAIN * side_deg))
        speed = COME_SPEED * min(1.0, (distance_cm - SAFE_DISTANCE_CM) / SLOW_DOWN_CM)
        speed = max(FOLLOW_MIN_SPEED, speed)
        check(self.misty.drive_time(linearVelocity=round(speed), angularVelocity=round(turn),
                                    timeMs=FOLLOW_DRIVE_MS))

    def loop(self):
        face_tracking = self.face_tracking
        while self.running:
            time.sleep(FACE_CHECK_SEC)
            frame = self.perception.last_frame
            if frame is None or time.monotonic() < self.paused_until:
                continue
            try:
                faces, target, distance_cm = self.recognizer.find_closest(frame)
            except Exception as error:
                print(f"  Face tracking error: {error}")
                continue
            command = None
            try:
                with self.lock:
                    now = time.monotonic()
                    if now < self.paused_until:
                        continue  # a gesture reaction started while we were looking
                    if target:
                        self.last_face_time = now
                    if 0 < self.following_until <= now:
                        self.stop_following(f"gave up after {FOLLOW_MAX_SEC} s")
                    elif self.following() and not self.sees_face():
                        self.stop_following("lost sight of your face")
                    following = self.following()
                    close_enough = 0 < distance_cm <= face_tracking.TRACKING_DISTANCE_THRESHOLD_CM
                    if target and (following or close_enough):
                        if following:
                            self.drive_toward(target, distance_cm, frame.shape[1])
                        command = self.head.compute(target.bbox, frame.shape[1], frame.shape[0])
                        if command and self.following():
                            self.head.yaw = 0  # the body does the turning; the head only tilts
                            command = (0, command[1])
            except requests.RequestException as error:
                print(f"  Misty didn't respond: {error}")
            if command is None:
                continue
            yaw, pitch = command
            try:
                check(self.misty.move_head(pitch=pitch, roll=0, yaw=yaw,
                                           velocity=face_tracking.HEAD_VELOCITY, units="degrees"))
            except requests.RequestException as error:
                print(f"  Misty didn't respond: {error}")
            time.sleep(face_tracking.SETTLE_SECONDS)  # let the head stop before looking again

    def close(self):
        self.running = False
        self.thread.join(timeout=2.0)


class MoveInProgress:
    """Remembers the STOP / COME / GO_AWAY Misty is still doing, so a WAVE can't interrupt it.
    GO_AWAY always finishes: a STOP during it is ignored too."""

    MOVE_SEC = {"STOP": STOP_FREEZE_SEC, "COME": FOLLOW_MAX_SEC, "GO_AWAY": GO_AWAY_TIME_MS / 1000}

    def __init__(self):
        self.move = None
        self.until = 0.0

    def end(self, move, now, after=0.0):
        """The move turned out shorter (e.g. COME reached the person): it ends `after` seconds from now."""
        if self.move == move:
            self.until = now + after

    def allows(self, event, now):
        """Return False if this gesture must be ignored right now.
        Also notes when STOP, COME or GO_AWAY starts (a new one replaces the old one,
        so STOP cuts a COME short)."""
        busy = now < self.until
        if event == "WAVE":
            return not busy
        if event == "STOP" and busy and self.move == "GO_AWAY":
            return False
        if event in self.MOVE_SEC:
            self.move, self.until = event, now + self.MOVE_SEC[event]
        return True


def react(misty, event, follower=None):
    """Send Misty the reaction for one gesture. Returns True if COME is following the person's face."""
    if event in ("STOP", "COME", "GO_AWAY"):
        cancel_turn()
    follow = event == "COME" and follower is not None and follower.sees_face()
    if follower:
        # Gesture reactions move the head, so face tracking waits until they're done.
        # Except COME: if Misty sees the person's face, she drives toward it, stopping at a safe distance.
        if event == "STOP":
            follower.pause(STOP_FREEZE_SEC)
        elif follow:
            follower.start_following()
        elif event == "COME":
            follower.pause(COME_TIME_MS / 1000, yaw=0, pitch=HEAD_UP)
        elif event == "GO_AWAY":
            follower.pause(GO_AWAY_TIME_MS / 1000)  # she turns away, so there's no face to follow
    if event == "STOP":
        check(misty.stop())
        check(misty.halt())  # also stops the head mid-tilt
    elif event == "WAVE":
        greet(misty)
    elif event == "COME" and follow:
        check(misty.change_led(red=0, green=0, blue=255))  # the follower does the driving
    elif event == "COME":
        come_forward(misty)
    elif event == "GO_AWAY":
        go_away(misty)
    return follow


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
    parser.add_argument("--no-faces", action="store_true", help="don't turn Misty's head toward faces")
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
    follower = None
    if misty and not args.no_faces:
        print("Loading face recognition...")
        face_tracking = load_face_tracking()
        if face_tracking:
            follower = FaceFollower(misty, perception, face_tracking)
    print("Ready. Show a gesture (STOP, WAVE, COME, GO_AWAY) or pet Misty's head/chin.")
    print("Keys in the video window: h = head pat, c = chin pat, q = quit.")
    in_progress = MoveInProgress()
    try:
        while perception.last_key != "q":
            event = perception.poll()
            if follower and follower.follow_finished:
                follower.follow_finished = False
                in_progress.end("COME", time.monotonic())  # she reached you early, so WAVE works again
            if perception.last_key == "h":
                pet.handle_touch("head")
            elif perception.last_key == "c":
                pet.handle_touch("chin")
            if not event:
                continue
            if not in_progress.allows(event, time.monotonic()):
                print(f"{event} ignored: {in_progress.move} is still in progress")
                continue
            print(event)
            if misty:
                try:
                    following = react(misty, event, follower)
                    if event == "COME" and not following:
                        in_progress.end("COME", time.monotonic(), COME_TIME_MS / 1000)  # the fixed drive is shorter
                except requests.RequestException as error:
                    print(f"  Misty didn't respond: {error}")  # keep going, don't crash the demo
    finally:
        if follower:
            follower.close()
        perception.close()
        cancel_turn()
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
