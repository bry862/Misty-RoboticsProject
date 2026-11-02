from pet_personality import PetPersonality

class PetController:
    def __init__(self, misty=None):
        self.misty = misty
        self.personality = PetPersonality()

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


if __name__ == "__main__":
    controller = PetController()

    controller.handle_touch("head")
    controller.handle_touch("head")
