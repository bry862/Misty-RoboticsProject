"""Tests for gesture_tracking.perception using made-up hands

Run from the project folder with:  python -m unittest
"""
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from gesture_tracking import perception as p

FPS = 30
WIDTH, HEIGHT = 640, 480


def fake_hand(dx=0, bend=1.0, lean=0, fingers_open=True):
    """21 (x, y) pixel points for an upright hand with spread fingers.

    dx moves the hand sideways, bend < 1 tips it toward the person (it looks
    shorter on camera), lean pushes the fingers sideways, and fingers_open=False
    curls the fingers into a fist.
    """
    points = [(300 + dx, 300)] * 21
    points[p.WRIST] = (300 + dx, 400)
    for n, (tip, joint) in enumerate(p.FINGERS):
        x = 300 + dx - 30 + 20 * n
        points[tip - 3] = (x + lean * 0.5, 300)  # finger base
        points[joint] = (x + lean * 0.7, 250)
        points[tip] = (x + lean, 200 if fingers_open else 330)
    return [(x, 400 - (400 - y) * bend) for x, y in points]


def as_landmarks(points):
    """Turn pixel points into MediaPipe-style landmarks (fractions of the frame)."""
    return [SimpleNamespace(x=x / WIDTH, y=y / HEIGHT) for x, y in points]


def waving(t):
    return fake_hand(dx=60 * math.sin(2 * math.pi * 1.5 * t))


def upside_down(points):
    """The same hand turned upside down (rotated half a turn around the wrist): fingers point down."""
    return [(600 - x, 800 - y) for x, y in points]


def beckoning(t):
    """A hand bending at the wrist and back with the fingers up: a beckon (COME)."""
    return fake_hand(bend=0.7 + 0.3 * math.cos(2 * math.pi * 1.5 * t))


def shooing(t):
    """The same bend with the fingers pointing down: a shoo (GO_AWAY)."""
    return upside_down(beckoning(t))


BACK, PALM = "Right", "Left"


def turn_squash(t):
    """How wide the turning hand looks: 1 = back toward camera, 0 = edge-on, -1 = palm."""
    return math.cos(math.pi * (1 - math.cos(2 * math.pi * t)) / 2)


def turning_over(t):
    """A raised hand turning over (back -> edge-on -> palm) and back, once a second.

    Turning squashes the hand sideways, and past edge-on its left and right swap.
    """
    squash = turn_squash(t)
    wrist_x = fake_hand()[p.WRIST][0]
    return [(wrist_x + (x - wrist_x) * squash, y) for x, y in fake_hand()]


def moving_through(*keyframes):
    """A hand that slides smoothly between (time, dx) keyframes, then stays put."""
    def hand_at(t):
        for (t0, x0), (t1, x1) in zip(keyframes, keyframes[1:]):
            if t0 <= t <= t1:
                f = (1 - math.cos(math.pi * (t - t0) / (t1 - t0))) / 2  # ease in and out
                return fake_hand(dx=x0 + (x1 - x0) * f)
        return fake_hand(dx=keyframes[-1][1])
    return hand_at


def run_detector(detector, hand_at, seconds, *extra):
    """Feed a detector one frame at a time; return the times (s) it fired.

    extra is passed on to update() after the hand and time (e.g. the palm side).
    """
    fired = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        if detector.update(hand_at(t), t, *extra):
            fired.append(t)
    return fired


class FakeTracker:
    """Stands in for HandTracker: returns a black frame plus made-up hands.

    label is the left/right label given to every hand, or a function of time returning it.
    """

    def __init__(self, hands_at, clock, label):
        self.hands_at = hands_at
        self.clock = clock
        self.label = label
        self.closed = False

    def read(self):
        t = self.clock[0]
        label = self.label(t) if callable(self.label) else self.label
        hands = self.hands_at(t)
        return np.zeros((HEIGHT, WIDTH, 3), np.uint8), hands, [label] * len(hands)

    def close(self):
        self.closed = True


def run_perception(hands_at, seconds, label=PALM):
    """Run GesturePerception on made-up hands; return (perception, [(event, time)])."""
    clock = [0.0]
    perception = p.GesturePerception(show=False, tracker=FakeTracker(hands_at, clock, label))
    events = []
    with mock.patch.object(p.time, "monotonic", lambda: clock[0]):
        for i in range(int(seconds * FPS)):
            clock[0] = i / FPS
            event = perception.poll()
            if event:
                events.append((event, clock[0]))
    return perception, events


class TestHandShape(unittest.TestCase):
    def test_open_hand(self):
        self.assertTrue(p.is_open_hand(fake_hand()))

    def test_fist_is_not_open(self):
        self.assertFalse(p.is_open_hand(fake_hand(fingers_open=False)))

    def test_upright_hand_is_raised(self):
        self.assertTrue(p.is_raised(fake_hand()))

    def test_leaning_hand_is_not_raised(self):
        self.assertFalse(p.is_raised(fake_hand(lean=200)))

    def test_bending_shortens_reach(self):
        self.assertLess(p.finger_reach(fake_hand(bend=0.5)), p.finger_reach(fake_hand()) - 1)

    def test_raised_fingers_point_up(self):
        self.assertFalse(p.fingers_point_down(fake_hand()))

    def test_upside_down_fingers_point_down(self):
        self.assertTrue(p.fingers_point_down(upside_down(fake_hand())))

    def test_pick_closest_hand_takes_the_biggest(self):
        near = as_landmarks(fake_hand())
        far = [SimpleNamespace(x=lm.x * 0.3, y=lm.y * 0.3) for lm in near]
        self.assertIs(p.pick_closest_hand([far, near]), near)

    def test_pick_closest_hand_with_no_hands(self):
        self.assertIsNone(p.pick_closest_hand([]))


class TestPalmSide(unittest.TestCase):
    def test_right_hand_back_toward_camera(self):
        self.assertFalse(p.palm_faces_camera(fake_hand(), BACK))

    def test_left_hand_palm_toward_camera(self):
        self.assertTrue(p.palm_faces_camera(fake_hand(), PALM))

    def test_turning_the_hand_around_flips_it(self):
        turned = [(600 - x, y) for x, y in fake_hand()]  # same hand, seen from the other side
        self.assertTrue(p.palm_faces_camera(turned, BACK))

    def test_upside_down_hand_keeps_its_side(self):
        upside_down = [(600 - x, 600 - y) for x, y in fake_hand()]  # rotated 180°, not flipped
        self.assertFalse(p.palm_faces_camera(upside_down, BACK))

    def test_unknown_label(self):
        self.assertIsNone(p.palm_faces_camera(fake_hand(), None))


class TestCountSwings(unittest.TestCase):
    def test_counts_direction_changes(self):
        self.assertEqual(p.count_swings([0, 50, 0, 50, 0], min_move=30), 3)

    def test_ignores_small_moves(self):
        self.assertEqual(p.count_swings([0, 10, 0, 10, 0], min_move=30), 0)


class TestStop(unittest.TestCase):
    def test_held_palm_fires_once(self):
        fired = run_detector(p.StopDetector(), lambda t: fake_hand(), 2)
        self.assertEqual(len(fired), 1)
        self.assertAlmostEqual(fired[0], p.STOP_HOLD_SEC, delta=0.05)

    def test_short_hold_does_not_fire(self):
        self.assertEqual(run_detector(p.StopDetector(), lambda t: fake_hand(), 0.3), [])

    def test_fires_again_after_hand_drops(self):
        hand_at = lambda t: None if 0.7 <= t < 1.0 else fake_hand()
        self.assertEqual(len(run_detector(p.StopDetector(), hand_at, 2)), 2)

    def test_fist_does_not_fire(self):
        self.assertEqual(run_detector(p.StopDetector(), lambda t: fake_hand(fingers_open=False), 2), [])

    def test_wave_does_not_fire(self):
        self.assertEqual(run_detector(p.StopDetector(), waving, 2), [])

    def test_palm_toward_misty_fires(self):
        self.assertEqual(len(run_detector(p.StopDetector(), lambda t: fake_hand(), 2, True)), 1)

    def test_back_of_hand_does_not_fire(self):
        # Palm facing the person, e.g. a hand left raised after a COME beckon (found in testing).
        self.assertEqual(run_detector(p.StopDetector(), lambda t: fake_hand(), 2, False), [])


class TestWave(unittest.TestCase):
    def test_wave_fires(self):
        self.assertEqual(len(run_detector(p.WaveDetector(), waving, 1.5)), 1)

    def test_cooldown_between_waves(self):
        fired = run_detector(p.WaveDetector(), waving, 6)
        gaps = [b - a for a, b in zip(fired, fired[1:])]
        self.assertTrue(all(gap >= p.GESTURE_COOLDOWN_SEC for gap in gaps))

    def test_still_palm_does_not_fire(self):
        self.assertEqual(run_detector(p.WaveDetector(), lambda t: fake_hand(), 3), [])

    def test_small_jitter_does_not_fire(self):
        jitter = lambda t: fake_hand(dx=10 * math.sin(9.4 * t))
        self.assertEqual(run_detector(p.WaveDetector(), jitter, 3), [])

    def test_moving_fist_does_not_fire(self):
        fist = lambda t: fake_hand(dx=60 * math.sin(9.4 * t), fingers_open=False)
        self.assertEqual(run_detector(p.WaveDetector(), fist, 3), [])

    def test_hand_passing_through_does_not_fire(self):
        sliding = lambda t: fake_hand(dx=-200 + 150 * t)
        self.assertEqual(run_detector(p.WaveDetector(), sliding, 3), [])

    def test_repositioning_hand_does_not_fire(self):
        # Move to a new spot, overshoot, settle back, small adjustment (found on the robot)
        reposition = moving_through((0, 0), (0.3, 100), (0.5, 60), (0.7, 95), (1.0, 95))
        self.assertEqual(run_detector(p.WaveDetector(), reposition, 3), [])

    def test_small_adjustments_do_not_fire(self):
        adjusting = moving_through((0, 0), (0.25, 40), (0.5, 0), (0.75, 40), (1.0, 40))
        self.assertEqual(run_detector(p.WaveDetector(), adjusting, 3), [])

    def test_slow_wave_fires(self):
        slow = lambda t: fake_hand(dx=60 * math.sin(2 * math.pi * 1.0 * t))
        self.assertEqual(len(run_detector(p.WaveDetector(), slow, 2)), 1)

    def test_small_wave_fires(self):
        small = lambda t: fake_hand(dx=40 * math.sin(2 * math.pi * 1.5 * t))
        self.assertEqual(len(run_detector(p.WaveDetector(), small, 2)), 1)

    def test_wave_with_palm_toward_misty_fires(self):
        self.assertEqual(len(run_detector(p.WaveDetector(), waving, 1.5, True)), 1)


class TestWaveIgnoresOtherMoves(unittest.TestCase):
    """Pieces of other gestures shouldn't add up to a wave (found on the robot: COME then STOP gave WAVE)."""

    def test_turning_hand_around_does_not_fire(self):
        # Each swing comes with the hand turning to show the other side.
        detector = p.WaveDetector()
        fired = [t for i in range(3 * FPS) for t in [i / FPS]
                 if detector.update(waving(t), t, math.cos(2 * math.pi * 1.5 * t) > 0)]
        self.assertEqual(fired, [])

    def test_one_frame_on_the_wrong_side_does_not_break_a_wave(self):
        detector = p.WaveDetector()
        fired = [t for i in range(int(1.5 * FPS)) for t in [i / FPS]
                 if detector.update(waving(t), t, i % 10 != 5)]  # every 10th frame looks turned
        self.assertEqual(len(fired), 1)

    def test_swings_split_by_a_closed_hand_do_not_add_up(self):
        # Half a second open, half a second curled (like the fingers during a beckon).
        pieces = lambda t: fake_hand(dx=60 * math.sin(2 * math.pi * 1.5 * t), fingers_open=int(t * 2) % 2 == 0)
        self.assertEqual(run_detector(p.WaveDetector(), pieces, 4), [])


class TestCome(unittest.TestCase):
    def test_beckon_fires(self):
        self.assertEqual(len(run_detector(p.ComeDetector(), beckoning, 1.5)), 1)

    def test_shoo_does_not_fire(self):
        self.assertEqual(run_detector(p.ComeDetector(), shooing, 3), [])

    def test_small_bends_do_not_fire(self):
        noise = lambda t: fake_hand(bend=0.95 + 0.05 * math.cos(9.4 * t))
        self.assertEqual(run_detector(p.ComeDetector(), noise, 3), [])

    def test_wave_does_not_fire(self):
        self.assertEqual(run_detector(p.ComeDetector(), waving, 3), [])

    def test_hand_passing_through_does_not_fire(self):
        sliding = lambda t: fake_hand(dx=-200 + 150 * t)
        self.assertEqual(run_detector(p.ComeDetector(), sliding, 3), [])

    def test_turning_the_hand_over_does_not_fire(self):
        # Found in testing: seen edge-on the hand looks very long for its width, like a big bend.
        self.assertEqual(run_detector(p.ComeDetector(), turning_over, 4), [])


class TestGoAway(unittest.TestCase):
    def test_shoo_fires(self):
        self.assertEqual(len(run_detector(p.GoAwayDetector(), shooing, 1.5)), 1)

    def test_beckon_does_not_fire(self):
        self.assertEqual(run_detector(p.GoAwayDetector(), beckoning, 3), [])

    def test_cooldown_between_shoos(self):
        fired = run_detector(p.GoAwayDetector(), shooing, 6)
        gaps = [b - a for a, b in zip(fired, fired[1:])]
        self.assertTrue(all(gap >= p.GESTURE_COOLDOWN_SEC for gap in gaps))

    def test_upside_down_wave_does_not_fire(self):
        self.assertEqual(run_detector(p.GoAwayDetector(), lambda t: upside_down(waving(t)), 3), [])

    def test_turning_the_hand_over_does_not_fire(self):
        upside_down_turn = lambda t: upside_down(turning_over(t))
        self.assertEqual(run_detector(p.GoAwayDetector(), upside_down_turn, 4), [])

    def test_fingers_flipping_up_and_down_does_not_fire(self):
        # Up one frame, down the next: neither direction wins, so neither gesture fires.
        flipping = lambda t: shooing(t) if round(t * FPS) % 2 else beckoning(t)
        self.assertEqual(run_detector(p.GoAwayDetector(), flipping, 3), [])
        self.assertEqual(run_detector(p.ComeDetector(), flipping, 3), [])


class TestGesturePerception(unittest.TestCase):
    def test_still_palm_gives_stop(self):
        perception, events = run_perception(lambda t: [as_landmarks(fake_hand())], 2)
        self.assertEqual([e for e, _ in events], ["STOP"])
        self.assertEqual(perception.last_event, "STOP")
        self.assertAlmostEqual(perception.last_event_time, events[0][1])

    def test_wave_gives_wave(self):
        _, events = run_perception(lambda t: [as_landmarks(waving(t))], 1.5)
        self.assertEqual([e for e, _ in events], ["WAVE"])

    def test_beckon_gives_come(self):
        _, events = run_perception(lambda t: [as_landmarks(beckoning(t))], 1.5)
        self.assertEqual([e for e, _ in events], ["COME"])

    def test_shoo_gives_go_away(self):
        _, events = run_perception(lambda t: [as_landmarks(shooing(t))], 1.5)
        self.assertEqual([e for e, _ in events], ["GO_AWAY"])

    def test_palm_side_does_not_matter_for_come_and_go_away(self):
        for label in (BACK, PALM, None):
            _, come = run_perception(lambda t: [as_landmarks(beckoning(t))], 1.5, label)
            _, go_away = run_perception(lambda t: [as_landmarks(shooing(t))], 1.5, label)
            self.assertEqual([e for e, _ in come], ["COME"])
            self.assertEqual([e for e, _ in go_away], ["GO_AWAY"])

    def test_still_palm_toward_misty_gives_only_stop(self):
        _, events = run_perception(lambda t: [as_landmarks(fake_hand())], 2, PALM)
        self.assertEqual([e for e, _ in events], ["STOP"])

    def test_come_then_turning_to_stop_gives_no_wave(self):
        # Beckon, then turn the hand around a few times on the way to a STOP palm.
        def hands_at(t):
            if t < 1.2:
                return [as_landmarks(beckoning(t))]
            if t < 2.6:
                return [as_landmarks(waving(t))]
            return [as_landmarks(fake_hand())]
        def label_at(t):
            if t < 1.2:
                return BACK
            if t < 2.6:
                return PALM if math.cos(2 * math.pi * 1.5 * t) > 0 else BACK
            return PALM
        _, events = run_perception(hands_at, 4, label_at)
        self.assertEqual([e for e, _ in events], ["COME", "STOP"])

    def test_no_wave_right_after_another_gesture(self):
        hands_at = lambda t: [as_landmarks(beckoning(t) if t < 1.0 else waving(t))]
        _, events = run_perception(hands_at, 4, lambda t: BACK if t < 1.0 else PALM)
        (come, t_come), (wave, t_wave) = events[:2]
        self.assertEqual((come, wave), ("COME", "WAVE"))
        self.assertGreaterEqual(t_wave - t_come, p.AFTER_GESTURE_QUIET_SEC)

    def test_stop_is_not_held_back_after_another_gesture(self):
        hands_at = lambda t: [as_landmarks(shooing(t) if t < 1.0 else fake_hand())]
        _, events = run_perception(hands_at, 2, PALM)
        (shoo, t_shoo), (stop, t_stop) = events
        self.assertEqual((shoo, stop), ("GO_AWAY", "STOP"))
        self.assertLess(t_stop - t_shoo, p.AFTER_GESTURE_QUIET_SEC)

    def test_come_then_holding_the_hand_up_gives_no_stop(self):
        # Beckon, then leave the hand raised and still, back still toward Misty (found in testing).
        hands_at = lambda t: [as_landmarks(beckoning(t) if t < 1.2 else fake_hand())]
        _, events = run_perception(hands_at, 3, BACK)
        self.assertEqual([e for e, _ in events], ["COME"])

    def test_turning_the_hand_over_gives_no_come(self):
        # Turn the hand over and back a few times (found in testing: gave COME).
        # MediaPipe can switch its left/right guess as the hand turns over (a right
        # palm looks like the back of a left hand), so the palm side keeps reading "back".
        label_at = lambda t: BACK if turn_squash(t) > 0 else PALM
        _, events = run_perception(lambda t: [as_landmarks(turning_over(t))], 4, label_at)
        self.assertNotIn("COME", [e for e, _ in events])
        self.assertNotIn("GO_AWAY", [e for e, _ in events])

    def test_no_hands_gives_nothing(self):
        _, events = run_perception(lambda t: [], 2)
        self.assertEqual(events, [])

    def test_only_the_closest_hand_counts(self):
        def hands_at(t):
            near = as_landmarks(fake_hand())
            far_waving = [SimpleNamespace(x=lm.x * 0.3 + 0.6, y=lm.y * 0.3)
                          for lm in as_landmarks(waving(t))]
            return [far_waving, near]
        _, events = run_perception(hands_at, 1.5)
        self.assertEqual([e for e, _ in events], ["STOP"])

    def test_close_releases_the_camera(self):
        perception, _ = run_perception(lambda t: [], 0.1)
        perception.close()
        self.assertTrue(perception.tracker.closed)


if __name__ == "__main__":
    unittest.main()
