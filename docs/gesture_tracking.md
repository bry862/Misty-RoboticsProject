MediaPipe report 21 points on one hand:
index 0      = wrist
index 1–4    = thumb (base → tip)
index 5–8    = index finger (base → tip)
index 9–12   = middle finger
index 13–16  = ring finger
index 17–20  = pinky


python -m gesture_tracking.perception

python gesture_demo.py --webcam --no-robot




python gesture_demo.py --ip <misty-ip>            # the real demo, with Misty's camera
python gesture_demo.py --ip <misty-ip> --webcam   # backup: PC webcam, Misty still reacts

To run: 
source .venv/bin/activate
python gesture_demo.py --ip 192.168.1.5

