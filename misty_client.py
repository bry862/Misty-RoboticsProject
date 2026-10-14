
import requests
import json
import websocket


class MistyClient:

    #Store Misty's netwrok address
        #When we make a MistyClient object in the main.py
        #we will pass as a parameter the ip acdress. 
        #it should be 192.168.1.5
        #double checl the urls

    def __init__(self, ip):
        self.base_url = f"http://{ip}/api"
        self.websocket_url = f"ws://{ip}/pubsub"

    #Mehtods

    #Method to Move head
    def move_head(self, pitch, roll, yaw, velocity=100):
        response = requests.post(
        f"{self.base_url}/head",
        json={
            "Pitch": pitch,
            "Roll": roll,
            "Yaw": yaw,
            "Velocity": velocity
        },
        timeout=10
        )

        response.raise_for_status()

    #Method to move arm
    def move_arm(self, arm, position, velocity=100):
        response = requests.post(
            f"{self.base_url}/arms",
            json={
                "Arm": arm,
                "Position": position,
                "Velocity": velocity
            },
            timeout=10
        )

        response.raise_for_status()

    #Method to change face for greeting
    def display_image(self, fileName, alpha=1, layer=None, isURL=False):
        response = requests.post(
            f"{self.base_url}/images/display", 
            json={
                "FileName": fileName,
                "Alpha": alpha,
                "Layer": layer,
                "IsUrl": isURL
            }, 
            timeout=10
        )

        response.raise_for_status()
    


    #Method for speach
    def say(self, message):
        response = requests.post(
            f"{self.base_url}/tts/speak",
            json={"Text": message},
            timeout=10
        )
        response.raise_for_status()

    #Method for face recognition
    def start_face_recognition(self):

        print ("Start() called")
        response = requests.post(
            f"{self.base_url}/faces/recognition/start",
            timeout=10
        )
        response.raise_for_status()

    #Method to be alerted when a new face is encountered
        #For this, we need the faces we intend to identify to be 
        #scanned already using the built in FaceID feature. 
    def listen_for_faces(self, callback):

        print ("listen() called")

        def on_open(ws):

            print("WebSocket connected to Misty")
            subscription = {
                "Operation": "subscribe",
                "Type": "FaceRecognition",
                "DebounceMs": 200,
                "EventName": "FaceRecognition",
                "ReturnProperty": None,
                "EventConditions": []
            }

            ws.send(json.dumps(subscription))
            self.start_face_recognition()

        def on_message(ws, message):
            print("Message received:", message)

            data = json.loads(message)

            # Ignore subscription acknowledgements and other messages
            if data.get("eventName") != "FaceRecognition":
                return

            event_data = data.get("message", {})
            person_name = event_data.get("personName")

            if person_name and person_name != "unknown_person":
                callback(person_name)

        self.socket = websocket.WebSocketApp(
            self.websocket_url,
            on_open=on_open,
            on_message=on_message
        )

        self.socket.run_forever()
    

    #Gets called automatically when Websocket opens
    def on_open(ws):
        print("WebSocket connected to Misty")

        # Remove the previous subscription
        unsubscribe = {
            "Operation": "unsubscribe",
            "EventName": "FaceRecognition"
        }

        ws.send(json.dumps(unsubscribe))

        # Register the subscription again
        subscription = {
            "Operation": "subscribe",
            "Type": "FaceRecognition",
            "DebounceMs": 200,
            "EventName": "FaceRecognition",
            "ReturnProperty": None,
            "EventConditions": []
        }

        ws.send(json.dumps(subscription))

        self.start_face_recognition()
