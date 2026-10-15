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


def beckoning(t):
    return fake_hand(bend=0.7 + 0.3 * math.cos(2 * math.pi * 1.5 * t))


def run_detector(detector, hand_at, seconds):
    """Feed a detector one frame at a time; return the times (s) it fired."""
    fired = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        if detector.update(hand_at(t), t):
            fired.append(t)
    return fired


class FakeTracker:
    """Stands in for HandTracker: returns a black frame plus made-up hands."""

    def __init__(self, hands_at, clock):
        self.hands_at = hands_at
        self.clock = clock
        self.closed = False

    def read(self):
        return np.zeros((HEIGHT, WIDTH, 3), np.uint8), self.hands_at(self.clock[0])

    def close(self):
        self.closed = True


def run_perception(hands_at, seconds):
    """Run GesturePerception on made-up hands; return (perception, [(event, time)])."""
    clock = [0.0]
    perception = p.GesturePerception(show=False, tracker=FakeTracker(hands_at, clock))
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

    def test_pick_closest_hand_takes_the_biggest(self):
        near = as_landmarks(fake_hand())
        far = [SimpleNamespace(x=lm.x * 0.3, y=lm.y * 0.3) for lm in near]
        self.assertIs(p.pick_closest_hand([far, near]), near)

    def test_pick_closest_hand_with_no_hands(self):
        self.assertIsNone(p.pick_closest_hand([]))


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


class TestCome(unittest.TestCase):
    def test_beckon_fires(self):
        self.assertEqual(len(run_detector(p.ComeDetector(), beckoning, 1.5)), 1)

    def test_small_bends_do_not_fire(self):
        noise = lambda t: fake_hand(bend=0.95 + 0.05 * math.cos(9.4 * t))
        self.assertEqual(run_detector(p.ComeDetector(), noise, 3), [])

    def test_wave_does_not_fire(self):
        self.assertEqual(run_detector(p.ComeDetector(), waving, 3), [])

    def test_hand_passing_through_does_not_fire(self):
        sliding = lambda t: fake_hand(dx=-200 + 150 * t)
        self.assertEqual(run_detector(p.ComeDetector(), sliding, 3), [])


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
        self.assertTrue(perception._tracker.closed)


if __name__ == "__main__":
    unittest.main()
