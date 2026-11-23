"""LLM-backed reactions for PetPersonality.

get_reaction() returns a {"message", "emotion", "action"} dict in the same shape
as the presets, or None on any failure so the caller can use a preset instead.
"""
import json

# Only names the presets already use, so Misty is never asked for something
# she can't do.
ALLOWED_EMOTIONS = {"happy", "playful", "curious"}
ALLOWED_ACTIONS = {"hug", "love", "cute", "admire", "admire2", "surprise", "surprise2"}
MAX_MESSAGE_WORDS = 12

SYSTEM_PROMPT = f"""You are Misty, a small friendly robot pet being touched by a person.
Reply with ONLY a JSON object: {{"message": str, "emotion": str, "action": str}}
- message: what you say out loud, at most {MAX_MESSAGE_WORDS} words, warm and playful
- emotion: one of {sorted(ALLOWED_EMOTIONS)}
- action: one of {sorted(ALLOWED_ACTIONS)}
Never repeat your previous message."""


class OllamaClient:
    """Minimal client for a local Ollama server (no API key needed)."""

    def __init__(self, model="llama3.2", url="http://localhost:11434"):
        self.model = model
        self.url = url

    def chat(self, system, user, timeout):
        import requests  # imported here so tests don't need it installed

        response = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": "json",
                "stream": False,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class LLMResponder:
    def __init__(self, client=None, timeout=1.5):
        # client is anything with chat(system, user, timeout) -> str
        self.client = client or OllamaClient()
        self.timeout = timeout

    def get_reaction(self, location, context=None):
        context = context or {}
        user = f"You were just touched on your {location}."
        if context.get("head_pat_count"):
            user += f" That is head pat number {context['head_pat_count']} in a row."
        if context.get("last_message"):
            user += f' Your previous message was: "{context["last_message"]}".'

        try:
            raw = self.client.chat(SYSTEM_PROMPT, user, self.timeout)
            reaction = json.loads(raw)
        except Exception:
            return None

        if not self._valid(reaction, context.get("last_message")):
            return None

        return {
            "message": reaction["message"].strip(),
            "emotion": reaction["emotion"],
            "action": reaction["action"],
        }

    @staticmethod
    def _valid(reaction, last_message=None):
        if not isinstance(reaction, dict):
            return False
        message = reaction.get("message")
        if not isinstance(message, str) or not message.strip():
            return False
        if len(message.split()) > MAX_MESSAGE_WORDS:
            return False
        if message.strip() == last_message:
            return False
        return (
            reaction.get("emotion") in ALLOWED_EMOTIONS
            and reaction.get("action") in ALLOWED_ACTIONS
        )
