
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import cv2

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))


def get_embedding(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(image_path)

    faces = app.get(image)

    if len(faces) != 1:
        raise ValueError(
            f"Expected 1 face in {image_path}, found {len(faces)}"
        )

    return faces[0].normed_embedding


# Generate embeddings
person_a1 = get_embedding("person_a1.jpg")
person_a2 = get_embedding("person_a2.jpg")
person_b = get_embedding("person_b.png")

# Compare embeddings
same_person = cosine_similarity(
    [person_a1], [person_a2]
)[0][0]

different_people = cosine_similarity(
    [person_a1], [person_b]
)[0][0]

print(f"Same person similarity: {same_person:.4f}")
print(f"Different people similarity: {different_people:.4f}")
