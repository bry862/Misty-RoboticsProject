from pet_personality import PetPersonality
from llm_responder import LLMResponder
import time
import requests

class PetController:
    def __init__(self, misty=None, use_llm=False):
        self.misty = misty
        #use_llm needs a local Ollama server; if it is unreachable,
        #the personality silently falls back to its preset reactions
        responder = LLMResponder() if use_llm else None
        self.personality = PetPersonality(responder=responder)

    def handle_touch(self, location):
        reaction = self.personality.react_to_touch(location)

        if reaction is None:
            print("Touch ignored (cooldown or unknown location).")
            return

        print(f"Touch detected: {location}")
        print(f"Emotion: {reaction['emotion']}")
        print(f"Misty says: {reaction['message']}")

        #Simulation mode
        if self.misty is None:
            return

        #Send commands using MistyClient
        self.misty.say(reaction["message"])

        if reaction["emotion"] == "happy":
            self.misty.display_image("e_Joy.jpg")
            self.misty.move_head(-10, 0, 0, 50)

        elif reaction["emotion"] == "curious":
            self.misty.display_image("e_Admiration.jpg")
            self.misty.move_head(0, 15, 0, 40)
        
        elif reaction["emotion"] == "excited":
            response = requests.post(
                f"{self.misty.base_url}/actions/start",
                json={"Name": "walk-happy"},
                timeout=10
            )
            response.raise_for_status()
            print("Happy dance triggered!")

            #Dance finishes --> reset body to neutral
            time.sleep(2)

            response = requests.post(
                f"{self.misty.base_url}/actions/start",
                json={"Name": "body-reset"},
                timeout=5
            )
            response.raise_for_status()
            print("Misty returned to neutral!")
  


if __name__ == "__main__":
    import sys

    controller = PetController(use_llm="--llm" in sys.argv)

    controller.handle_touch("head")
    controller.handle_touch("head")
