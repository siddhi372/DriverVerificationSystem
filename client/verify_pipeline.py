import cv2
import numpy as np
import onnxruntime as ort
import json
import os
import time
import requests
import time
from datetime import datetime

from insightface.app import FaceAnalysis


# =========================================================
# 1. FACE DETECTION - YUNET
# =========================================================

FACE_MODEL = "models/face_detection_yunet_2026may.onnx"

detector = cv2.FaceDetectorYN.create(
    FACE_MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)


# =========================================================
# 2. LIVENESS MODEL
# =========================================================

LIVENESS_MODEL = "models/MiniFASNetV2.onnx"

session = ort.InferenceSession(
    LIVENESS_MODEL,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Liveness model loaded")


# =========================================================
# 3. FACE RECOGNITION MODEL
# =========================================================

recognition_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

recognition_app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

print("Face recognition model loaded")


# =========================================================
# 4. LOAD ALL REGISTERED DRIVERS
# =========================================================

DRIVER_FOLDER = "data/drivers"

registered_drivers = []


if not os.path.exists(DRIVER_FOLDER):

    print("ERROR: data/drivers folder not found.")
    exit()


for filename in os.listdir(DRIVER_FOLDER):

    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(
        DRIVER_FOLDER,
        filename
    )

    try:

        with open(file_path, "r") as file:
            driver_data = json.load(file)

        embedding = np.array(
            driver_data["embedding"],
            dtype=np.float32
        )

        norm = np.linalg.norm(embedding)

        if norm == 0:
            print(
                f"Skipping {filename}: invalid embedding."
            )
            continue

        embedding = embedding / norm

        registered_drivers.append({
            "driver_id": driver_data["driver_id"],
            "driver_name": driver_data["driver_name"],
            "embedding": embedding
        })

        print(
            f"Loaded driver: "
            f"{driver_data['driver_id']} - "
            f"{driver_data['driver_name']}"
        )

    except Exception as e:

        print(
            f"Could not load {filename}: {e}"
        )


if len(registered_drivers) == 0:

    print("ERROR: No registered drivers found.")
    exit()


print(
    f"Total registered drivers: "
    f"{len(registered_drivers)}"
)


# =========================================================
# 5. AGE ESTIMATION MODEL
# =========================================================

AGE_MODEL = (
    r"C:\Users\Siddhi\.insightface\models"
    r"\buffalo_l\genderage.onnx"
)

age_session = ort.InferenceSession(
    AGE_MODEL,
    providers=["CPUExecutionProvider"]
)

age_input_name = age_session.get_inputs()[0].name
age_output_name = age_session.get_outputs()[0].name

print("Age model loaded")

# =========================================================
# BACKEND API
# =========================================================

API_URL = "http://127.0.0.1:8000/verification/result"

session_id = datetime.now().strftime(
    "SESSION_%Y%m%d_%H%M%S"
)

last_sent_decision = None


def send_verification_result(
    decision,
    vehicle_status,
    liveness_status,
    liveness_confidence,
    driver_id=None,
    driver_name="UNKNOWN",
    driver_confidence=0.0,
    matched=False,
    estimated_age=None,
    processing_time=0.0
):

    global last_sent_decision

    # Prevent sending the same final result every camera frame
    if last_sent_decision == decision:
        return

    data = {

        "timestamp":
            datetime.now().isoformat(),

        "session_id":
            session_id,

        "face_detection": {

            "detected": True,

            "face_count": 1

        },

        "image_quality": {

            "status":
                "GOOD"

        },

        "liveness": {

            "status":
                liveness_status,

            "confidence":
                float(liveness_confidence)

        },

        "driver_authentication": {

            "driver_id":
                driver_id,

            "driver_name":
                driver_name,

            "confidence":
                float(driver_confidence),

            "matched":
                bool(matched)

        },

        "estimated_age":
            estimated_age,

        "decision": {

            "result":
                decision,

            "vehicle_status":
                vehicle_status

        },

        "processing_time":
            float(processing_time)

    }

    try:

        response = requests.post(
            API_URL,
            json=data,
            timeout=5
        )

        if response.status_code == 200:

            print(
                "Backend result sent successfully:"
            )

            print(
                response.json()
            )

            last_sent_decision = decision

        else:

            print(
                "Backend error:",
                response.status_code,
                response.text
            )

    except requests.RequestException as e:

        print(
            "Could not connect to backend:",
            e
        )
# =========================================================
# 6. FACE QUALITY
# =========================================================

def check_face_quality(frame, face, landmarks):

    height, width = frame.shape[:2]

    x, y, w, h = face[:4].astype(int)


    # -----------------------------------------
    # Face outside camera
    # -----------------------------------------

    if (
        x < 0
        or y < 0
        or x + w > width
        or y + h > height
    ):

        return False, "Make sure your full face is visible"


    # -----------------------------------------
    # Face too small
    # -----------------------------------------

    if w < 150 or h < 150:

        return False, "Move closer to the camera"


    # -----------------------------------------
    # Face crop
    # -----------------------------------------

    face_crop = frame[
        y:y + h,
        x:x + w
    ]


    if face_crop.size == 0:

        return False, "Please show your face"


    # -----------------------------------------
    # Brightness
    # -----------------------------------------

    gray = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray)


    if brightness < 45:

        return False, "Please stand in better lighting"


    # -----------------------------------------
    # Face direction
    # -----------------------------------------

    left_eye = np.array([
        landmarks[0],
        landmarks[1]
    ])

    nose = np.array(
        landmarks[2]
    )

    eye_center = np.mean(
        left_eye,
        axis=0
    )

    eye_distance = abs(
        landmarks[1][0]
        -
        landmarks[0][0]
    )


    if eye_distance > 0:

        nose_offset = abs(
            nose[0]
            -
            eye_center[0]
        )

        ratio = (
            nose_offset
            /
            eye_distance
        )


        if ratio > 0.45:

            return False, "Please look straight at the camera"


    return True, "Face position OK"


# =========================================================
# 7. FACE ALIGNMENT
# =========================================================

def align_face(face, landmarks):

    target = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)


    landmarks = np.array(
        landmarks,
        dtype=np.float32
    )


    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks,
        target,
        method=cv2.LMEDS
    )


    if matrix is None:

        return None


    aligned = cv2.warpAffine(
        face,
        matrix,
        (112, 112)
    )


    return aligned


# =========================================================
# 8. LIVENESS
# =========================================================

def check_liveness(face):

    # -----------------------------------------
    # Resize
    # -----------------------------------------

    face = cv2.resize(
        face,
        (80, 80)
    )


    # -----------------------------------------
    # Float32
    # -----------------------------------------

    face = face.astype(
        np.float32
    )


    # -----------------------------------------
    # HWC -> CHW
    # -----------------------------------------

    face = np.transpose(
        face,
        (2, 0, 1)
    )


    # -----------------------------------------
    # Add batch dimension
    # -----------------------------------------

    face = np.expand_dims(
        face,
        axis=0
    )


    # -----------------------------------------
    # Model inference
    # -----------------------------------------

    output = session.run(
        [output_name],
        {
            input_name: face
        }
    )[0]


    # -----------------------------------------
    # Softmax
    # -----------------------------------------

    probabilities = np.exp(
        output
        -
        np.max(
            output,
            axis=1,
            keepdims=True
        )
    )


    probabilities = (
        probabilities
        /
        probabilities.sum(
            axis=1,
            keepdims=True
        )
    )


    probabilities = probabilities[0]


    # -----------------------------------------
    # Debug output
    # -----------------------------------------

    print(
        "Liveness probabilities:",
        probabilities
    )


    # -----------------------------------------
    # Prediction
    # -----------------------------------------

    prediction = int(
        np.argmax(probabilities)
    )


    confidence = float(
        probabilities[prediction]
    )


    # -----------------------------------------
    # Class mapping
    #
    # Class 1 = LIVE
    # Class 0/2 = SPOOF
    #
    # Confirmed using your real-face
    # and phone-image tests.
    # -----------------------------------------

    if prediction == 1:

        label = "LIVE"

    else:

        label = "SPOOF"


    return label, confidence


# =========================================================
# 9. COSINE SIMILARITY
# =========================================================

def cosine_similarity(
    embedding1,
    embedding2
):

    embedding1 = (
        embedding1
        /
        np.linalg.norm(embedding1)
    )


    embedding2 = (
        embedding2
        /
        np.linalg.norm(embedding2)
    )


    similarity = np.dot(
        embedding1,
        embedding2
    )


    return float(similarity)


# =========================================================
# 10. FACE RECOGNITION
# =========================================================

def verify_driver(face_image):

    faces = recognition_app.get(
        face_image
    )


    # -----------------------------------------
    # No face
    # -----------------------------------------

    if len(faces) == 0:

        return "UNKNOWN", 0.0, None


    # -----------------------------------------
    # Get first face
    # -----------------------------------------

    face = faces[0]

    current_embedding = face.embedding


    if current_embedding is None:

        return "UNKNOWN", 0.0, None


    # -----------------------------------------
    # Normalize embedding
    # -----------------------------------------

    current_embedding = np.asarray(
        current_embedding,
        dtype=np.float32
    )


    norm = np.linalg.norm(
        current_embedding
    )


    if norm == 0:

        return "UNKNOWN", 0.0, None


    current_embedding = (
        current_embedding
        /
        norm
    )


    # -----------------------------------------
    # Compare against ALL drivers
    # -----------------------------------------

    best_similarity = -1.0
    best_driver = None


    for driver in registered_drivers:

        similarity = cosine_similarity(
            driver["embedding"],
            current_embedding
        )


        print(
            f'{driver["driver_name"]}: '
            f'{similarity:.3f}'
        )


        if similarity > best_similarity:

            best_similarity = similarity
            best_driver = driver


    # -----------------------------------------
    # Recognition threshold
    # -----------------------------------------

    threshold = 0.40


    if (
        best_driver is not None
        and best_similarity >= threshold
    ):

        return (
            "MATCH",
            best_similarity,
            best_driver
        )


    return (
        "UNKNOWN",
        best_similarity,
        None
    )


# =========================================================
# 11. AGE ESTIMATION
# =========================================================

def estimate_age(face_image):

    # -----------------------------------------
    # Resize to 96 x 96
    # -----------------------------------------

    face_image = cv2.resize(
        face_image,
        (96, 96)
    )


    # -----------------------------------------
    # BGR -> RGB
    # -----------------------------------------

    face_image = cv2.cvtColor(
        face_image,
        cv2.COLOR_BGR2RGB
    )


    # -----------------------------------------
    # Float32
    # -----------------------------------------

    face_image = face_image.astype(
        np.float32
    )


    # -----------------------------------------
    # HWC -> CHW
    # -----------------------------------------

    face_image = np.transpose(
        face_image,
        (2, 0, 1)
    )


    # -----------------------------------------
    # Batch dimension
    # -----------------------------------------

    face_image = np.expand_dims(
        face_image,
        axis=0
    )


    # -----------------------------------------
    # Model inference
    # -----------------------------------------

    output = age_session.run(
        [age_output_name],
        {
            age_input_name: face_image
        }
    )[0][0]


    # -----------------------------------------
    # Age calculation
    # -----------------------------------------

    age = int(
        round(output[2] * 100)
    )


    print(
        "Estimated age:",
        age
    )


    return age


# =========================================================
# 12. CAMERA SETTINGS
# =========================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


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

    print("ERROR: Could not access camera.")
    exit()


# =========================================================
# 13. LIVENESS STABILITY
# =========================================================

LIVE_REQUIRED_FRAMES = 5
SPOOF_REQUIRED_FRAMES = 3

live_count = 0
spoof_count = 0


# =========================================================
# 14. START
# =========================================================

print(
    "Starting Driver Verification Pipeline"
)

print(
    "Press Q to exit"
)


# =========================================================
# 15. MAIN LOOP
# =========================================================

while True:
    verification_start_time = time.time()
    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break

    height, width = frame.shape[:2]

    # -----------------------------------------
    # Set detector size
    # -----------------------------------------

    detector.setInputSize(
        (width, height)
    )

    # -----------------------------------------
    # Detect faces
    # -----------------------------------------

    _, faces = detector.detect(
        frame
    )

    # =====================================================
    # NO FACE
    # =====================================================

    if faces is None or len(faces) == 0:
        live_count = 0
        spoof_count = 0

        cv2.putText(
            frame,
            "Please show your face",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # =====================================================
    # MULTIPLE FACES
    # =====================================================

    elif len(faces) > 1:
        live_count = 0
        spoof_count = 0

        cv2.putText(
            frame,
            "Only one person should be in the camera",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # =====================================================
    # ONE FACE
    # =====================================================

    else:
        face = faces[0]

        x, y, w, h = face[:4].astype(
            int
        )

        # -------------------------------------------------
        # YuNet landmarks
        # -------------------------------------------------

        landmarks = np.array([
            [face[4], face[5]],
            [face[6], face[7]],
            [face[8], face[9]],
            [face[10], face[11]],
            [face[12], face[13]]
        ], dtype=np.float32)

        # -------------------------------------------------
        # Face quality
        # -------------------------------------------------

        quality_ok, message = check_face_quality(
            frame,
            face,
            landmarks
        )

        if not quality_ok:
            live_count = 0
            spoof_count = 0

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

        else:
            aligned_face = align_face(
                frame,
                landmarks
            )

            if aligned_face is None:
                live_count = 0
                spoof_count = 0

                cv2.putText(
                    frame,
                    "Face alignment failed",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            else:
                label, confidence = check_liveness(
                    aligned_face
                )

                if label == "LIVE":
                    live_count += 1
                    spoof_count = 0
                else:
                    spoof_count += 1
                    live_count = 0

                if live_count >= LIVE_REQUIRED_FRAMES:
                    stable_label = "LIVE"
                elif spoof_count >= SPOOF_REQUIRED_FRAMES:
                    stable_label = "SPOOF"
                else:
                    stable_label = "CHECKING"

                if stable_label == "SPOOF":
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"SPOOF {confidence:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Spoof detected - Access denied",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    processing_time = time.time() - verification_start_time

                    send_verification_result(
                        decision="DENY",
                        vehicle_status="LOCKED",
                        liveness_status="SPOOF",
                        liveness_confidence=confidence,
                        driver_id=None,
                        driver_name="UNKNOWN",
                        driver_confidence=0.0,
                        matched=False,
                        estimated_age=None,
                        processing_time=processing_time
                    )

                elif stable_label == "LIVE":
                    result, similarity, matched_driver = verify_driver(
                        aligned_face
                    )

                    estimated_age = estimate_age(
                        aligned_face
                    )

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"LIVE {confidence:.2f}",
                        (x, y - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{result} {similarity:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    if result == "MATCH":
                        driver_name = matched_driver["driver_name"]
                        driver_id = matched_driver["driver_id"]

                        cv2.putText(
                            frame,
                            f"{driver_name} verified",
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Estimated Age: {estimated_age}",
                            (30, 85),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                        processing_time = time.time() - verification_start_time

                        send_verification_result(
                            decision="VERIFIED",
                            vehicle_status="UNLOCKED",
                            liveness_status="LIVE",
                            liveness_confidence=confidence,
                            driver_id=driver_id,
                            driver_name=driver_name,
                            driver_confidence=similarity,
                            matched=True,
                            estimated_age=estimated_age,
                            processing_time=processing_time
                        )

                    else:
                        cv2.putText(
                            frame,
                            "Unknown driver - Access denied",
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Estimated Age: {estimated_age}",
                            (30, 85),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                        processing_time = time.time() - verification_start_time
                        send_verification_result(
                            decision="DENY",
                                    vehicle_status="LOCKED",
                                    liveness_status="LIVE",
                                    liveness_confidence=confidence,
                                    driver_id=None,
                                    driver_name="UNKNOWN",
                                    driver_confidence=similarity,
                                    matched=False,
                                    estimated_age=estimated_age,
                                    processing_time=processing_time
                                )
        

                else:
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Checking liveness...",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

# =========================================================
# 16. DISPLAY
# =========================================================

    cv2.imshow(
        "Driver Verification Pipeline",
        frame
    )

    # =====================================================
    # PRESS Q TO EXIT
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# 17. RELEASE RESOURCES
# =========================================================

camera.release()

cv2.destroyAllWindows()