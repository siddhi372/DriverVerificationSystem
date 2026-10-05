import cv2
import numpy as np
import onnxruntime as ort


# ==========================================
# 1. FACE QUALITY CHECK
# ==========================================

def check_face_quality(frame, face, landmarks):

    height, width = frame.shape[:2]

    x, y, w, h = face[:4].astype(int)

    # --------------------------------------
    # Face partially outside camera
    # --------------------------------------

    if x < 0 or y < 0 or x + w > width or y + h > height:
        return False, "Make sure your full face is visible"

    # --------------------------------------
    # Face too small
    # --------------------------------------

    if w < 150 or h < 150:
        return False, "Move closer to the camera"

    # --------------------------------------
    # Check brightness
    # --------------------------------------

    face_crop = frame[y:y+h, x:x+w]

    if face_crop.size == 0:
        return False, "Please show your face"

    gray = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray)

    if brightness < 45:
        return False, "Please stand in better lighting"

    # --------------------------------------
    # Check face orientation
    # --------------------------------------

    left_eye = np.array(landmarks[0])
    right_eye = np.array(landmarks[1])
    nose = np.array(landmarks[2])

    eye_center = (left_eye + right_eye) / 2

    eye_distance = abs(
        right_eye[0] - left_eye[0]
    )

    if eye_distance > 0:

        nose_offset = abs(
            nose[0] - eye_center[0]
        )

        ratio = nose_offset / eye_distance

        if ratio > 0.45:
            return False, "Please look straight at the camera"

    return True, "Face position OK"


# ==========================================
# 2. FACE DETECTION - YuNet
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
# 3. LIVENESS MODEL
# ==========================================

LIVENESS_MODEL = "models/MiniFASNetV2.onnx"

session = ort.InferenceSession(
    LIVENESS_MODEL,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Liveness model loaded")


# ==========================================
# 4. 2.7x FACE CROP
# ==========================================

def crop_face(frame, box, scale=2.7):

    x, y, w, h = box

    height, width = frame.shape[:2]

    scale = min(
        (height - 1) / h,
        (width - 1) / w,
        scale
    )

    new_w = w * scale
    new_h = h * scale

    center_x = x + w / 2
    center_y = y + h / 2

    x1 = max(
        0,
        int(center_x - new_w / 2)
    )

    y1 = max(
        0,
        int(center_y - new_h / 2)
    )

    x2 = min(
        width - 1,
        int(center_x + new_w / 2)
    )

    y2 = min(
        height - 1,
        int(center_y + new_h / 2)
    )

    cropped = frame[
        y1:y2 + 1,
        x1:x2 + 1
    ]

    return cropped


# ==========================================
# 5. LIVENESS PREDICTION
# ==========================================

def check_liveness(face):

    # MiniFASNetV2 input = 80 x 80

    face = cv2.resize(
        face,
        (80, 80)
    )

    # IMPORTANT:
    # Do NOT divide by 255

    face = face.astype(
        np.float32
    )

    # HWC -> CHW

    face = np.transpose(
        face,
        (2, 0, 1)
    )

    # Add batch dimension

    face = np.expand_dims(
        face,
        axis=0
    )

    # Model inference

    output = session.run(
        [output_name],
        {
            input_name: face
        }
    )[0]

    # --------------------------------------
    # Softmax
    # --------------------------------------

    probabilities = np.exp(
        output -
        np.max(
            output,
            axis=1,
            keepdims=True
        )
    )

    probabilities = probabilities / probabilities.sum(
        axis=1,
        keepdims=True
    )

    probabilities = probabilities[0]

    print(
        "Probabilities:",
        probabilities
    )

    # --------------------------------------
    # MiniFASNetV2 classes
    #
    # 0 = Fake
    # 1 = Real
    # 2 = Fake
    # --------------------------------------

    prediction = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[prediction]
    )

    if prediction == 1:
        label = "LIVE"
    else:
        label = "SPOOF"

    return label, confidence


# ==========================================
# 6. CAMERA
# ==========================================

camera = cv2.VideoCapture(0)


while True:

    success, frame = camera.read()

    if not success:

        print("Could not access camera")

        break


    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(
        frame
    )


    # ======================================
    # CASE 1: NO FACE
    # ======================================

    if faces is None or len(faces) == 0:

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
    # CASE 2: MORE THAN ONE FACE
    # ======================================

    elif len(faces) > 1:

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
    # CASE 3: EXACTLY ONE FACE
    # ======================================

    else:

        face = faces[0]

        x, y, w, h = face[:4].astype(int)


        # ----------------------------------
        # Get 5 facial landmarks
        # ----------------------------------

        landmarks = np.array([
            [face[4], face[5]],     # eye 1
            [face[6], face[7]],     # eye 2
            [face[8], face[9]],     # nose
            [face[10], face[11]],   # mouth 1
            [face[12], face[13]]    # mouth 2
        ], dtype=np.float32)


        # ----------------------------------
        # Check face quality
        # ----------------------------------

        quality_ok, message = check_face_quality(
            frame,
            face,
            landmarks
        )


        # ==================================
        # BAD FACE QUALITY
        # ==================================

        if not quality_ok:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                message,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


        # ==================================
        # GOOD FACE QUALITY
        # ==================================

        else:

            # --------------------------------
            # Create expanded face crop
            # --------------------------------

            face_crop = crop_face(
                frame,
                (x, y, w, h)
            )


            if face_crop.size != 0:

                # ----------------------------
                # Run liveness ONCE
                # ----------------------------

                label, confidence = check_liveness(
                    face_crop
                )

                text = f"{label} {confidence:.2f}"


                # ----------------------------
                # Draw face box
                # ----------------------------

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )


                # ----------------------------
                # Display liveness
                # ----------------------------

                cv2.putText(
                    frame,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        "Driver Liveness Detection",
        frame
    )


    # ======================================
    # PRESS Q TO EXIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()

cv2.destroyAllWindows()