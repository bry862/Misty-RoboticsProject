from pet_personality import PetPersonality
import time

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

        #Send commands using the SDK robot inside MistyClient
        robot = self.misty.robot
        robot.speak(text=reaction["message"])

        if reaction["emotion"] == "happy":
            robot.display_image(fileName="e_Joy.jpg")
            robot.move_head(pitch=-10, roll=0, yaw=0, velocity=50)

        elif reaction["emotion"] == "curious":
            robot.display_image(fileName="e_Admiration.jpg")
            robot.move_head(pitch=0, roll=15, yaw=0, velocity=40)

        elif reaction["emotion"] == "excited":
            response = robot.start_action(name="walk-happy")
            response.raise_for_status()
            print("Happy dance triggered!")

            #Dance finishes --> reset body to neutral
            time.sleep(2)

            response = robot.start_action(name="body-reset")
            response.raise_for_status()
            print("Misty returned to neutral!")
  


if __name__ == "__main__":
    controller = PetController()

    controller.handle_touch("head")
    controller.handle_touch("head")
