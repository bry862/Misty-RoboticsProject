#misty_client.py

from mistyPy.Robot import Robot
from mistyPy.Events import Events

class MistyClient:
    def __init__(self, ip_address: str):
        self.robot = Robot(ip_address)

    def start_face_recognition(self, callback_function):
        self.robot.start_face_recognition()
        self.robot.register_event(
            event_name="FaceRecognitionEvent",
            event_type=Events.FaceRecognition,
            callback_function=callback_function,
            keep_alive=True
        )

    def stop_face_recognition(self):
        self.robot.stop_face_recognition()
        self.robot.unregister_event("FaceRecognitionEvent")

    def take_picture_base64(self):
        """Requests a base64 image string from Misty's camera and parses the JSON response."""
        try:
            response = self.robot.take_picture(
                fileName="current_face.jpg",
                width=640,
                height=480,
                displayOnScreen=False,
                overwriteExisting=True,
                base64=True
            )
            if hasattr(response, 'json'):
                return response.json()
            return response
        except Exception as e:
            print(f"Error capturing picture from robot: {e}")
            return None

    def move_head_safely(self, pitch=0, yaw=0, velocity=30):
        """Moves Misty's head with safety clamping bounds for absolute positions."""
        try:
            # Clamp absolute angles to safe mechanical boundaries for Misty II
            safe_yaw = float(max(-40, min(40, yaw)))
            safe_pitch = float(max(-25, min(25, pitch)))
            safe_vel = int(max(10, min(100, velocity)))

            self.robot.move_head(roll=0, pitch=safe_pitch, yaw=safe_yaw, velocity=safe_vel)
        except Exception as e:
            print(f"Error moving head: {e}")

     # ---------- Object / person detection ----------
    def start_object_detection(self, callback_function, min_confidence=0.5,
                               debounce=500):
        self.robot.start_object_detector(min_confidence, 0, 15)
        self.robot.register_event(
            event_name="ObjectDetectionEvent",
            event_type=Events.ObjectDetection,
            callback_function=callback_function,
            debounce=debounce,
            keep_alive=True,
        )

    def stop_object_detection(self):
        self.robot.stop_object_detector()
        self.robot.unregister_event("ObjectDetectionEvent")

        # ---------- Driving ----------
    
    
    ##### Not uded Yet #####
    def drive(self, linear_velocity=0, angular_velocity=0):
        try:
            self.robot.drive(int(linear_velocity), int(angular_velocity))
        except Exception as e:
            print(f"Error driving: {e}")

    def stop(self):
        try:
            self.robot.stop()
        except Exception as e:
            print(f"Error stopping: {e}")

     # ---------- Lifecycle ----------
    def keep_alive(self):
        self.robot.keep_alive()

    def shutdown(self):
        """Safe cleanup: stop motion and tear down events."""
        self.stop()
        for stopper in (self.stop_object_detection,
                        self.stop_face_recognition):
            try:
                stopper()
            except Exception:
                pass