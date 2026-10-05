import cv2
import numpy as np
import json
from insightface.app import FaceAnalysis


# ==========================================
# 1. LOAD ARCFACE
# ==========================================

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# ==========================================
# 2. LOAD REGISTERED DRIVER
# ==========================================

DRIVER_FILE = "data/drivers/DRV001.json"

with open(DRIVER_FILE, "r") as file:
    driver_data = json.load(file)


registered_embedding = np.array(
    driver_data["embedding"],
    dtype=np.float32
)

driver_name = driver_data["driver_name"]
driver_id = driver_data["driver_id"]


# ==========================================
# 3. COSINE SIMILARITY
# ==========================================

def cosine_similarity(embedding1, embedding2):

    embedding1 = embedding1 / np.linalg.norm(
        embedding1
    )

    embedding2 = embedding2 / np.linalg.norm(
        embedding2
    )

    similarity = np.dot(
        embedding1,
        embedding2
    )

    return float(similarity)


# ==========================================
# 4. CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

print("Starting driver verification...")
print("Press Q to exit.")


while True:

    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break


    # ======================================
    # FACE DETECTION + EMBEDDING
    # ======================================

    faces = app.get(frame)


    # --------------------------------------
    # NO FACE
    # --------------------------------------

    if len(faces) == 0:

        cv2.putText(
            frame,
            "Please show your face",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # --------------------------------------
    # MULTIPLE FACES
    # --------------------------------------

    elif len(faces) > 1:

        cv2.putText(
            frame,
            "Only one person allowed",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # --------------------------------------
    # ONE FACE
    # --------------------------------------

    else:

        face = faces[0]

        # Get current face embedding

        current_embedding = face.embedding


        # Normalize

        current_embedding = (
            current_embedding /
            np.linalg.norm(current_embedding)
        )


        # ==================================
        # COMPARE
        # ==================================

        similarity = cosine_similarity(
            registered_embedding,
            current_embedding
        )


        print(
            "Similarity:",
            similarity
        )


        # ==================================
        # DECISION
        # ==================================

        # Initial prototype threshold
        threshold = 0.40


        if similarity >= threshold:

            result = "MATCH"
            message = f"{driver_name} verified"

        else:

            result = "UNKNOWN"
            message = "Unknown driver"


        # ==================================
        # DISPLAY
        # ==================================

        x1, y1, x2, y2 = (
            face.bbox.astype(int)
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            result,
            (x1, y1 - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Similarity: {similarity:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            message,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    cv2.imshow(
        "Driver Verification",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()