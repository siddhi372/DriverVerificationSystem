import cv2
import numpy as np

from insightface.app import FaceAnalysis


# =====================================================
# FACE RECOGNITION MODEL
# =====================================================

recognition_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

recognition_app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

print("AI Enrollment Service: Face model loaded")


# =====================================================
# GENERATE FACE EMBEDDING
# =====================================================

def generate_face_embedding(image_bytes: bytes):

    # -------------------------------------------------
    # Convert uploaded bytes to numpy array
    # -------------------------------------------------

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )


    # -------------------------------------------------
    # Decode image
    # -------------------------------------------------

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if frame is None:

        raise ValueError(
            "Unable to read enrollment image"
        )


    # -------------------------------------------------
    # Detect faces
    # -------------------------------------------------

    faces = recognition_app.get(
        frame
    )


    # -------------------------------------------------
    # No face
    # -------------------------------------------------

    if len(faces) == 0:

        raise ValueError(
            "No face detected. Please position the driver's face clearly."
        )


    # -------------------------------------------------
    # Multiple faces
    # -------------------------------------------------

    if len(faces) > 1:

        raise ValueError(
            "Multiple faces detected. Only the driver should be visible."
        )


    # -------------------------------------------------
    # Get driver's face
    # -------------------------------------------------

    face = faces[0]


    # -------------------------------------------------
    # Get ArcFace embedding
    # -------------------------------------------------

    embedding = face.embedding


    if embedding is None:

        raise ValueError(
            "Unable to generate face embedding"
        )


    # -------------------------------------------------
    # Convert to float32
    # -------------------------------------------------

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )


    # -------------------------------------------------
    # Normalize
    # -------------------------------------------------

    norm = np.linalg.norm(
        embedding
    )


    if norm == 0:

        raise ValueError(
            "Invalid face embedding"
        )


    embedding = (
        embedding / norm
    )


    # -------------------------------------------------
    # Convert to Python list
    # -------------------------------------------------

    embedding = embedding.tolist()


    # -------------------------------------------------
    # Return AI result
    # -------------------------------------------------

    return {
        "embedding": embedding,
        "embedding_dimension": len(
            embedding
        ),
        "face_count": len(faces)
    }