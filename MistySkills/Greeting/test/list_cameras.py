import cv2

for i in range(10):
    camera = cv2.VideoCapture(i)

    if camera.isOpened():
        success, frame = camera.read()

        if success:
            height, width = frame.shape[:2]
            print(f"Camera {i}: {width}x{height}")

        camera.release()