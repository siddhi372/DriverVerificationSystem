import cv2
import numpy as np
import json
import os

from insightface.app import FaceAnalysis


# =========================================================
# SETTINGS
# =========================================================

TOTAL_SAMPLES = 20

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

DETECTION_SIZE = (320, 320)


# =========================================================
# REGISTRATION POSITIONS
# =========================================================

POSITIONS = [
    ("FRONT", 4),
    ("LEFT", 4),
    ("RIGHT", 4),
    ("UP", 4),
    ("DOWN", 4),
]


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading face recognition model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=DETECTION_SIZE
)

print("Model loaded.")


# =========================================================
# DRIVER DETAILS
# =========================================================

driver_id = input("Enter Driver ID: ").strip()

driver_name = input("Enter Driver Name: ").strip()


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

camera.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not camera.isOpened():

    print("ERROR: Could not open camera.")

    exit()


# =========================================================
# VARIABLES
# =========================================================

embeddings = []

current_position_index = 0

position_sample_count = 0

total_captured = 0


# =========================================================
# CURRENT POSITION
# =========================================================

current_position, required_samples = POSITIONS[
    current_position_index
]


# =========================================================
# WINDOW
# =========================================================

WINDOW_NAME = "Driver Face Registration"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    CAMERA_WIDTH,
    CAMERA_HEIGHT
)


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    success, frame = camera.read()

    if not success:

        print("Camera frame error.")

        break


    # =====================================================
    # MIRROR CAMERA
    # =====================================================

    frame = cv2.flip(
        frame,
        1
    )


    # =====================================================
    # DISPLAY INFORMATION
    # =====================================================

    title = (
        f"LOOK {current_position}"
    )

    cv2.putText(
        frame,
        title,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Press SPACE: {position_sample_count}/{required_samples}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Total: {total_captured}/{TOTAL_SAMPLES}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # =====================================================
    # INSTRUCTIONS
    # =====================================================

    if current_position == "FRONT":

        instruction = (
            "Look straight at camera"
        )

    elif current_position == "LEFT":

        instruction = (
            "Turn your face LEFT"
        )

    elif current_position == "RIGHT":

        instruction = (
            "Turn your face RIGHT"
        )

    elif current_position == "UP":

        instruction = (
            "Look slightly UP"
        )

    else:

        instruction = (
            "Look slightly DOWN"
        )


    cv2.putText(
        frame,
        instruction,
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Q = Cancel",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        2
    )


    # =====================================================
    # SHOW CAMERA
    # =====================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # =====================================================
    # KEYBOARD
    # =====================================================

    key = cv2.waitKey(1) & 0xFF


    # =====================================================
    # QUIT
    # =====================================================

    if key == ord("q"):

        print("Registration cancelled.")

        break


    # =====================================================
    # CAPTURE WITH SPACE
    # =====================================================

    if key == 32:

        print(
            f"\nCapturing "
            f"{current_position} "
            f"sample "
            f"{position_sample_count + 1}/"
            f"{required_samples}"
        )


        # -------------------------------------------------
        # Detect face ONLY when SPACE is pressed
        # -------------------------------------------------

        faces = app.get(frame)


        # -------------------------------------------------
        # No face
        # -------------------------------------------------

        if len(faces) == 0:

            print(
                "No face detected. "
                "Try again."
            )

            continue


        # -------------------------------------------------
        # Multiple faces
        # -------------------------------------------------

        if len(faces) > 1:

            print(
                "Multiple faces detected. "
                "Only one person should be visible."
            )

            continue


        # -------------------------------------------------
        # Get face
        # -------------------------------------------------

        face = faces[0]


        # -------------------------------------------------
        # Check face size
        # -------------------------------------------------

        x1, y1, x2, y2 = face.bbox.astype(int)

        face_width = x2 - x1

        frame_width = frame.shape[1]

        face_ratio = (
            face_width /
            frame_width
        )


        if face_ratio < 0.10:

            print(
                "Face is too far. "
                "Move closer."
            )

            continue


        if face_ratio > 0.75:

            print(
                "Face is too close. "
                "Move back."
            )

            continue


        # -------------------------------------------------
        # Get embedding
        # -------------------------------------------------

        embedding = face.embedding


        if embedding is None:

            print(
                "Could not generate embedding."
            )

            continue


        # -------------------------------------------------
        # Normalize
        # -------------------------------------------------

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )


        norm = np.linalg.norm(
            embedding
        )


        if norm == 0:

            print(
                "Invalid embedding."
            )

            continue


        embedding = (
            embedding /
            norm
        )


        # -------------------------------------------------
        # Save sample
        # -------------------------------------------------

        embeddings.append(
            embedding.copy()
        )


        position_sample_count += 1

        total_captured += 1


        print(
            f"✓ Captured "
            f"{current_position} "
            f"{position_sample_count}/"
            f"{required_samples}"
        )


        # =================================================
        # POSITION COMPLETE
        # =================================================

        if position_sample_count >= required_samples:

            current_position_index += 1


            # ------------------------------------------------
            # ALL POSITIONS COMPLETE
            # ------------------------------------------------

            if (
                current_position_index
                >= len(POSITIONS)
            ):

                print(
                    "\nAll face samples captured!"
                )

                break


            # ------------------------------------------------
            # NEXT POSITION
            # ------------------------------------------------

            current_position, required_samples = POSITIONS[
                current_position_index
            ]

            position_sample_count = 0


            print(
                "\n================================"
            )

            print(
                f"Now look {current_position}"
            )

            print(
                f"Take {required_samples} samples"
            )

            print(
                "Press SPACE for each sample."
            )

            print(
                "================================"
            )


# =========================================================
# RELEASE CAMERA
# =========================================================

camera.release()

cv2.destroyAllWindows()


# =========================================================
# CHECK SAMPLES
# =========================================================

if len(embeddings) < TOTAL_SAMPLES:

    print(
        f"\nRegistration incomplete."
    )

    print(
        f"Captured: {len(embeddings)}/{TOTAL_SAMPLES}"
    )

    exit()


# =========================================================
# CREATE FINAL EMBEDDING
# =========================================================

print(
    "\nCreating final face embedding..."
)


embeddings_array = np.asarray(
    embeddings,
    dtype=np.float32
)


# Average all samples
final_embedding = np.mean(
    embeddings_array,
    axis=0
)


# Normalize final embedding
norm = np.linalg.norm(
    final_embedding
)


if norm == 0:

    print(
        "ERROR: Invalid final embedding."
    )

    exit()


final_embedding = (
    final_embedding /
    norm
)


# =========================================================
# SAVE DRIVER
# =========================================================

os.makedirs(
    "data/drivers",
    exist_ok=True
)


driver_data = {

    "driver_id": driver_id,

    "driver_name": driver_name,

    "embedding": final_embedding.tolist()

}


file_path = (
    f"data/drivers/{driver_id}.json"
)


with open(
    file_path,
    "w"
) as file:

    json.dump(
        driver_data,
        file,
        indent=4
    )


# =========================================================
# SUCCESS
# =========================================================

print(
    "\n========================================"
)

print(
    " DRIVER REGISTERED SUCCESSFULLY"
)

print(
    "========================================"
)

print(
    f"Driver ID   : {driver_id}"
)

print(
    f"Driver Name : {driver_name}"
)

print(
    f"Samples     : {len(embeddings)}"
)

print(
    f"Saved to    : {file_path}"
)

print(
    "========================================"
)