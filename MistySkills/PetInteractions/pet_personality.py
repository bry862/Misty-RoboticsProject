import random
import time

class PetPersonality:
    def __init__(self, cooldown=3):
        #Min seconds between reactions
        self.cooldown = cooldown
        self.last_reaction = 0

        #Possible reactions to being petted
        self.reactions = {
            "head": [
                "That feels nice!",
                "I love head pats!",
                "You're my favorite human!"
            ],
            "chin": [
                "Oh, hello there!",
                "That tickles!",
                "Are we playing?"
            ]
        }

    def react_to_touch(self, location):
        #Stop Misty from reacting too often
        current_time = time.monotonic()

        if current_time - self.last_reaction < self.cooldown:
            return None

        #Check whether we recognize the touch 
        if location not in self.reactions:
            return None

        #Random response
        message = random.choice(self.reactions[location])

        self.last_reaction = current_time

        return {
            "location": location,
            "message": message,
            "emotion": "happy" if location == "head" else "curious"
        }

if __name__ == "__main__":
    pet = PetPersonality()

    print("Testing head touch:")
    print(pet.react_to_touch("head"))

    print("Testing immediate second touch:")
    print(pet.react_to_touch("head"))

    time.sleep(3)

    print("Testing chin touch after cooldown:")
    print(pet.react_to_touch("chin"))
