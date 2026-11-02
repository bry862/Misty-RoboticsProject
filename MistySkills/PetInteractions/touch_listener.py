import argparse
import json
import websocket

from pet_controller import PetController


class TouchListener:
    def __init__(self, controller):
        self.controller = controller

    def handle_sensor(self, sensor, is_contacted):
        #Only react when a sensor is being touched
        if not is_contacted:
            return

        sensor = sensor.lower()

        if "chin" in sensor:
            self.controller.handle_touch("chin")
        elif any(part in sensor for part in
                 ("head", "front", "back", "left", "right", "scruff")):
            self.controller.handle_touch("head")

    def on_message(self, ws, message):
        data = json.loads(message)

        if data.get("eventName") != "TouchSensor":
            return

        event = data.get("message", {})
        sensor = event.get("sensorPosition", "")
        contacted = event.get("isContacted", False)

        self.handle_sensor(sensor, contacted)

    def on_open(self, ws):
        subscription = {
            "Operation": "subscribe",
            "Type": "TouchSensor",
            "EventName": "TouchSensor",
            "DebounceMs": 200
        }

        ws.send(json.dumps(subscription))
        print("Subscribed to Misty's touch sensors!")

    def listen(self, ip):
        url = f"ws://{ip}/pubsub"

        socket = websocket.WebSocketApp(
            url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=lambda ws, error: print("WebSocket error:", error),
            on_close=lambda ws, code, reason: print("Disconnected:", reason)
        )

        socket.run_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", help="Misty's IP address")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Test touch events without Misty"
    )
    args = parser.parse_args()

    if not args.simulate and not args.ip:
        parser.error("Provide --ip or use --simulate")

    #Connect to Misty when not in simulation mode
    if args.simulate:
        controller = PetController()
    else:
        from pathlib import Path
        import sys

        greeting_folder = Path(__file__).resolve().parents[1] / "Greeting"
        sys.path.insert(0, str(greeting_folder))

        from misty_client import MistyClient

        misty = MistyClient(args.ip)
        controller = PetController(misty)

    listener = TouchListener(controller)

    if args.simulate:
        print("Simulating head touch...")
        listener.handle_sensor("HeadFront", True)

        print("Simulating immediate second touch...")
        listener.handle_sensor("HeadFront", True)

        print("Simulating touch release...")
        listener.handle_sensor("HeadFront", False)
    else:
        print("Listening for Misty's touch sensors...")
        listener.listen(args.ip)
