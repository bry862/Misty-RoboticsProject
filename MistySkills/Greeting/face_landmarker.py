import mediapipe as mp
import cv2

# MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Configure the landmarker
options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_faces=5
)

# Load the model
with FaceLandmarker.create_from_options(options) as landmarker:

    image = cv2.imread("test.jpg")

    if image is None:
        raise FileNotFoundError("Could not load test.jpg")

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    # Detect faces and landmarks
    result = landmarker.detect(mp_image)

    print("Faces detected:", len(result.face_landmarks))

    for i, face in enumerate(result.face_landmarks):
        print(f"Face {i + 1}: {len(face)} landmarks")

    # Draw facial landmarks
for face in result.face_landmarks:
    for landmark in face:

        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])

        cv2.circle(
            image,
            (x, y),
            1,
            (0, 255, 0),
            -1
        )

cv2.imshow("Face Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()