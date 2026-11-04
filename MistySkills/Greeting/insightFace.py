
from insightface.app import FaceAnalysis
import cv2

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

# Load image
image = cv2.imread("test.jpg")

if image is None:
    raise FileNotFoundError("Could not load test.jpg")

# Analyze faces
faces = app.get(image)

print("Faces detected:", len(faces))

for i, face in enumerate(faces):
    embedding = face.embedding

    print(f"\nFace {i + 1}")
    print("Embedding dimensions:", embedding.shape)
    print("First 10 values:", embedding[:10])
