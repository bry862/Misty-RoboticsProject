import random
import time

class PetPersonality:
    def __init__(self, cooldown=3):
        #Min seconds between reactions
        self.cooldown = cooldown
        self.last_reaction = 0
        self.head_pat_count = 0
  
        #Possible reactions to being petted
        self.reactions = {
            "head": [
                {
                    "message": "That feels nice!",
                    "emotion": "happy",
                    "action": "hug"
                },
                {
                    "message": "I love head pats!",
                    "emotion": "happy",
                    "action": "love"
                },
                {
                    "message": "You are spoiling me!",
                    "emotion": "playful",
                    "action": "cute"
                },
                {
                    "message": "You're my favorite human!",
                    "emotion": "happy",
                    "action": "admire"
                },

            ],
            "chin": [
                {
                    "message": "Oh, hello there!",
                    "emotion": "curious",
                    "action": "admire2"
                },
                {
                    "message": "Oh, more attention for me?",
                    "emotion": "playful",
                    "action": "cute"
                },
                {
                    "message": "Ooh, what's that?",
                    "emotion": "curious",
                    "action": "surprise"
                },
                {
                    "message": "Hey! That tickles!",
                    "emotion": "playful",
                    "action": "surprise2"
                }
            ]
        }

        self.last_reaction_message = None


    def react_to_touch(self, location):
        #Stop Misty from reacting too often
        current_time = time.monotonic()

        if current_time - self.last_reaction < self.cooldown:
            return None

        #Check whether we recognize the touch 
        if location not in self.reactions:
            return None

        
        # Count head pats and trigger excitement every third pat
        if location == "head":
            self.head_pat_count += 1

            if self.head_pat_count >= 3:
                self.head_pat_count = 0
                self.last_reaction = current_time
                self.last_reaction_message = "You're making me so happy!"

                return {
                    "location": "head",
                    "message": "You're making me so happy!",
                    "emotion": "excited",
                    "action": "walk-happy"
                }

        
        #Don't repeat the same reaction twice in a row
        available = [
            reaction for reaction in self.reactions[location]
            if reaction["message"] != self.last_reaction_message
        ]

        reaction = random.choice(available)
        self.last_reaction_message = reaction["message"]
        self.last_reaction = current_time


        return {
            "location": location,
            "message": reaction["message"],
            "emotion": reaction["emotion"],
            "action": reaction["action"]
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
