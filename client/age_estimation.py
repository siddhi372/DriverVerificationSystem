import cv2
import numpy as np
import onnxruntime as ort


# ==========================================
# 1. YUNET FACE DETECTOR
# ==========================================

FACE_MODEL = "models/face_detection_yunet_2026may.onnx"

detector = cv2.FaceDetectorYN.create(
    FACE_MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)


# ==========================================
# 2. GENDER + AGE MODEL
# ==========================================

AGE_MODEL = r"C:\Users\Siddhi\.insightface\models\buffalo_l\genderage.onnx"

session = ort.InferenceSession(
    AGE_MODEL,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Age model loaded")

print(
    "Input shape:",
    session.get_inputs()[0].shape
)

print(
    "Output shape:",
    session.get_outputs()[0].shape
)


# ==========================================
# 3. AGE ESTIMATION
# ==========================================

def estimate_age(face_crop):

    # --------------------------------------
    # Resize to model input
    # --------------------------------------

    face_crop = cv2.resize(
        face_crop,
        (96, 96)
    )


    # --------------------------------------
    # BGR -> RGB
    # --------------------------------------

    face_crop = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------
    # Convert to float32
    # --------------------------------------

    face_crop = face_crop.astype(
        np.float32
    )


    # --------------------------------------
    # HWC -> CHW
    # --------------------------------------

    face_crop = np.transpose(
        face_crop,
        (2, 0, 1)
    )


    # --------------------------------------
    # Add batch dimension
    # --------------------------------------

    face_crop = np.expand_dims(
        face_crop,
        axis=0
    )


    # --------------------------------------
    # Model inference
    # --------------------------------------

    output = session.run(
        [output_name],
        {
            input_name: face_crop
        }
    )[0][0]


    # --------------------------------------
    # Convert model output to age
    # --------------------------------------

    age = int(
        round(
            output[2] * 100
        )
    )


    print(
        "Estimated age:",
        age
    )


    return age


# ==========================================
# 4. CAMERA
# ==========================================

camera = cv2.VideoCapture(0)


# Run age model only every 10 frames
frame_count = 0

# Store previous age
last_age = None


# ==========================================
# 5. MAIN LOOP
# ==========================================

while True:

    success, frame = camera.read()


    # --------------------------------------
    # Camera error
    # --------------------------------------

    if not success:

        print(
            "Could not access camera"
        )

        break


    frame_count += 1


    # --------------------------------------
    # Frame dimensions
    # --------------------------------------

    height, width = frame.shape[:2]


    # --------------------------------------
    # Update YuNet input size
    # --------------------------------------

    detector.setInputSize(
        (width, height)
    )


    # --------------------------------------
    # Detect faces
    # --------------------------------------

    _, faces = detector.detect(
        frame
    )


    # ======================================
    # 6. NO FACE
    # ======================================

    if faces is None or len(faces) == 0:

        # Reset previous age

        last_age = None


        cv2.putText(
            frame,
            "Please show your face",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ======================================
    # 7. MULTIPLE FACES
    # ======================================

    elif len(faces) > 1:

        # Reset previous age

        last_age = None


        cv2.putText(
            frame,
            "Only one person should be in the camera",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


    # ======================================
    # 8. ONE FACE
    # ======================================

    else:

        face = faces[0]


        # ----------------------------------
        # Face bounding box
        # ----------------------------------

        x, y, w, h = face[:4].astype(
            int
        )


        # ----------------------------------
        # Make coordinates valid
        # ----------------------------------

        x1 = max(
            0,
            x
        )

        y1 = max(
            0,
            y
        )

        x2 = min(
            width,
            x + w
        )

        y2 = min(
            height,
            y + h
        )


        # ----------------------------------
        # Extract face
        # ----------------------------------

        face_crop = frame[
            y1:y2,
            x1:x2
        ]


        # ==================================
        # 9. VALID FACE CROP
        # ==================================

        if face_crop.size != 0:


            # ------------------------------
            # Run age every 10 frames
            # ------------------------------

            if frame_count % 10 == 0:

                last_age = estimate_age(
                    face_crop
                )


            # ------------------------------
            # Draw face box
            # ------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ------------------------------
            # Display age
            # ------------------------------

            if last_age is not None:

                cv2.putText(
                    frame,
                    f"Estimated Age: {last_age}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


# ==========================================
# 10. DISPLAY CAMERA
# ==========================================

    cv2.imshow(
        "Age Estimation",
        frame
    )


    # ======================================
    # 11. PRESS Q TO EXIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# 12. RELEASE RESOURCES
# ==========================================

camera.release()

cv2.destroyAllWindows()