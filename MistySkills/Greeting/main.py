from misty_client import MistyClient

misty = MistyClient("192.168.1.5")
# test start


# misty.say("Hello! I am connected.")


# test end

greeted_people = set()
def greet_person(person_name):

    if person_name in greeted_people:
        return

    greeted_people.add(person_name)

    print(f"Recognized: {person_name}")

    misty.say(f"Hello, {person_name}!, how is your day going?")

    misty.move_head(0, 20, 0)

    misty.move_arm("left", -28, 50)

    misty.display_image("e_Admiration.jpg")

    
  


misty.listen_for_faces(greet_person)