
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

    #Commands


    #Command for speach
    def say(self, message):
        response = requests.post(
            f"{self.base_url}/tts/speak",
            json={"Text": message},
            timeout=10
        )
        response.raise_for_status()

    #Command for face recognition
    def start_face_recognition(self):
        response = requests.post(
            f"{self.base_url}/faces/recognition/start",
            timeout=10
        )
        response.raise_for_status()

    #Command to be alerted when a new face is encountered
        #For this, we need the faces we intend to identify to be 
        #scanned already using the built in FaceID feature. 
    def listen_for_faces(self, callback):
        def on_open(ws):
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
            data = json.loads(message)

            # Ignore subscription acknowledgements and other messages
            if data.get("eventName") != "FaceRecognition":
                return

            event_data = data.get("message", {})
            person_name = event_data.get("PersonName")

            if person_name and person_name != "unknown_person":
                callback(person_name)

        self.socket = websocket.WebSocketApp(
            self.websocket_url,
            on_open=on_open,
            on_message=on_message
        )

        self.socket.run_forever()
