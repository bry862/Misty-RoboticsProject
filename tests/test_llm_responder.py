"""Tests for PetInteractions.llm_responder and its fallback in PetPersonality.

Run from the project folder with:  python -m unittest
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "MistySkills", "PetInteractions"))

from llm_responder import LLMResponder  # noqa: E402
from pet_personality import PetPersonality  # noqa: E402


class FakeClient:
    def __init__(self, reply=None, error=None):
        self.reply, self.error = reply, error

    def chat(self, system, user, timeout):
        if self.error:
            raise self.error
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


GOOD = {"message": "Ooh, right there!", "emotion": "happy", "action": "love"}


class LLMResponderTests(unittest.TestCase):
    def test_valid_reply_matches_preset_shape(self):
        r = LLMResponder(FakeClient(GOOD)).get_reaction("head")
        self.assertEqual(r, GOOD)

    def test_invalid_action_returns_none(self):
        bad = {**GOOD, "action": "wiggle"}
        self.assertIsNone(LLMResponder(FakeClient(bad)).get_reaction("head"))

    def test_invalid_emotion_returns_none(self):
        bad = {**GOOD, "emotion": "furious"}
        self.assertIsNone(LLMResponder(FakeClient(bad)).get_reaction("head"))

    def test_too_long_message_returns_none(self):
        bad = {**GOOD, "message": "word " * 20}
        self.assertIsNone(LLMResponder(FakeClient(bad)).get_reaction("head"))

    def test_not_json_returns_none(self):
        self.assertIsNone(LLMResponder(FakeClient("hello")).get_reaction("head"))

    def test_client_error_returns_none(self):
        client = FakeClient(error=TimeoutError())
        self.assertIsNone(LLMResponder(client).get_reaction("head"))

    def test_repeating_last_message_returns_none(self):
        r = LLMResponder(FakeClient(GOOD)).get_reaction(
            "head", {"last_message": GOOD["message"]})
        self.assertIsNone(r)


class PersonalityFallbackTests(unittest.TestCase):
    def test_uses_llm_reaction(self):
        pet = PetPersonality(cooldown=0, responder=LLMResponder(FakeClient(GOOD)))
        r = pet.react_to_touch("chin")
        self.assertEqual(r["message"], GOOD["message"])
        self.assertEqual(r["location"], "chin")

    def test_falls_back_to_preset_on_failure(self):
        pet = PetPersonality(
            cooldown=0, responder=LLMResponder(FakeClient(error=TimeoutError())))
        r = pet.react_to_touch("chin")
        presets = [x["message"] for x in pet.reactions["chin"]]
        self.assertIn(r["message"], presets)

    def test_no_responder_still_works(self):
        r = PetPersonality(cooldown=0).react_to_touch("chin")
        self.assertIn("message", r)


if __name__ == "__main__":
    unittest.main()
