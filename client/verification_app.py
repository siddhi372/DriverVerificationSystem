import cv2
import time
import json
from datetime import datetime
import numpy as np
import onnxruntime as ort

from client.esp32_manager import ESP32Manager
from client.driver_verification import DriverVerifier
from api.api_client import send_verification_result


# =====================================================
# CONFIGURATION
# =====================================================
CAMERA_INDEX = 0
ESP32_PORT = "COM3"
BAUDRATE = 115200
TEST_MODE = True
PROCESSING_TIME = 5
RESULT_DISPLAY_TIME = 5

FACE_MODEL = "models/face_detection_yunet_2026may.onnx"
LIVENESS_MODEL = "models/MiniFASNetV2.onnx"
AGE_MODEL = r"C:\Users\Siddhi\.insightface\models\buffalo_l\genderage.onnx"

MIN_BRIGHTNESS = 45
MIN_FACE_WIDTH_RATIO = 0.10
MAX_FACE_WIDTH_RATIO = 0.75
RECOGNITION_THRESHOLD = 0.45
LIVE_REQUIRED_FRAMES = 5
SPOOF_REQUIRED_FRAMES = 3

WAITING = "WAITING"
VERIFYING = "VERIFYING"
RESULT = "RESULT"


# =====================================================
# ESP32
# =====================================================
esp32 = ESP32Manager(port=ESP32_PORT, baudrate=BAUDRATE)

if not TEST_MODE:
    if not esp32.connect():
        print("Could not connect to ESP32.")
        raise SystemExit


# =====================================================
# AI MODELS
# =====================================================
print("Loading driver verification...")
driver_verifier = DriverVerifier(
    driver_folder="data/drivers",
    threshold=RECOGNITION_THRESHOLD
)
print("Driver verification ready.")

print("Loading YuNet...")
face_detector = cv2.FaceDetectorYN.create(
    FACE_MODEL, "", (320, 320), 0.6, 0.3, 5000
)
print("YuNet loaded.")

print("Loading MiniFASNetV2...")
liveness_net = cv2.dnn.readNetFromONNX(LIVENESS_MODEL)
print("Liveness model loaded.")

print("Loading age model...")
try:
    age_session = ort.InferenceSession(
        AGE_MODEL,
        providers=["CPUExecutionProvider"]
    )
    age_input_name = age_session.get_inputs()[0].name
    age_output_name = age_session.get_outputs()[0].name
    AGE_MODEL_AVAILABLE = True
    print("Age model loaded.")
except Exception as e:
    age_session = None
    age_input_name = None
    age_output_name = None
    AGE_MODEL_AVAILABLE = False
    print("Age model unavailable:", e)


# =====================================================
# CAMERA
# =====================================================
print("Opening camera...")
camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    print("Could not access camera.")
    if not TEST_MODE:
        esp32.close()
    raise SystemExit


# =====================================================
# =====================================================
# WINDOW
# =====================================================
WINDOW_NAME = "Driver Verification System"

# Dashboard window size
DASHBOARD_W = 1500
DASHBOARD_H = 900

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DASHBOARD_W, DASHBOARD_H)


# =====================================================
# STATE
# =====================================================
state = WAITING
verification_start_time = None
result_start_time = None
session_id = None
result = None

faces = []
face = None
face_count = 0
detected_face = False
last_bounding_box = None

image_quality = "WAITING"
liveness = "WAITING"
liveness_confidence = 0.0

driver_id = None
driver_name = "UNKNOWN"
driver_confidence = 0.0
driver_matched = False
estimated_age = None

live_counter = 0
spoof_counter = 0
best_driver_id = None
best_driver_name = "UNKNOWN"
best_driver_confidence = 0.0
best_age = None


# =====================================================
# RESET
# =====================================================
def reset_verification():
    global faces, face, face_count, detected_face, last_bounding_box
    global image_quality, liveness, liveness_confidence
    global driver_id, driver_name, driver_confidence, driver_matched
    global estimated_age, live_counter, spoof_counter
    global best_driver_id, best_driver_name, best_driver_confidence, best_age

    faces = []
    face = None
    face_count = 0
    detected_face = False
    last_bounding_box = None

    image_quality = "CHECKING"
    liveness = "CHECKING"
    liveness_confidence = 0.0

    driver_id = None
    driver_name = "UNKNOWN"
    driver_confidence = 0.0
    driver_matched = False
    estimated_age = None

    live_counter = 0
    spoof_counter = 0
    best_driver_id = None
    best_driver_name = "UNKNOWN"
    best_driver_confidence = 0.0
    best_age = None


# =====================================================
# FACE CROP
# =====================================================
def crop_face(image, bbox, scale=1.5):
    image_h, image_w = image.shape[:2]
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None

    scale = min(scale, (image_h - 1) / h, (image_w - 1) / w)
    new_w = w * scale
    new_h = h * scale
    cx = x + w / 2
    cy = y + h / 2

    x1 = max(0, int(cx - new_w / 2))
    y1 = max(0, int(cy - new_h / 2))
    x2 = min(image_w - 1, int(cx + new_w / 2))
    y2 = min(image_h - 1, int(cy + new_h / 2))

    crop = image[y1:y2 + 1, x1:x2 + 1]
    return crop if crop.size else None


def softmax(scores):
    scores = scores - np.max(scores)
    p = np.exp(scores)
    total = np.sum(p)
    return p / total if total else np.zeros_like(p)


# =====================================================
# IMAGE QUALITY
# =====================================================
def check_image_quality(frame, detected_faces):
    if detected_faces is None or len(detected_faces) == 0:
        return False, "SHOW YOUR FACE"

    if len(detected_faces) > 1:
        return False, "ONLY ONE FACE ALLOWED"

    current = detected_faces[0]
    x, y, w, h = current[:4].astype(int)
    frame_h, frame_w = frame.shape[:2]

    if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        return False, "SHOW FULL FACE"

    crop = frame[max(0, y):min(frame_h, y + h), max(0, x):min(frame_w, x + w)]
    if crop.size == 0:
        return False, "SHOW YOUR FACE"

    brightness = float(np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
    if brightness < MIN_BRIGHTNESS:
        return False, "LIGHT TOO DIM"

    ratio = w / float(frame_w)
    if ratio < MIN_FACE_WIDTH_RATIO:
        return False, "COME CLOSER"
    if ratio > MAX_FACE_WIDTH_RATIO:
        return False, "MOVE BACK"

    return True, "GOOD"


# =====================================================
# LIVENESS
# =====================================================
def check_liveness(frame, current_face):
    if current_face is None:
        return "NO FACE", 0.0

    x, y, w, h = current_face[:4].astype(int)
    crop = crop_face(frame, (x, y, w, h), scale=2.7)
    if crop is None:
        return "UNKNOWN", 0.0

    crop = cv2.resize(crop, (80, 80)).astype(np.float32)
    blob = np.transpose(crop, (2, 0, 1))[None, ...]

    liveness_net.setInput(blob)
    output = liveness_net.forward()
    probabilities = softmax(np.asarray(output[0], dtype=np.float32).reshape(-1))
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])

    return ("LIVE" if prediction == 1 else "SPOOF"), confidence


# =====================================================
# AGE
# =====================================================
def estimate_age(face_image):
    if not AGE_MODEL_AVAILABLE or face_image is None or face_image.size == 0:
        return None

    try:
        image = cv2.resize(face_image, (96, 96))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1))[None, ...]
        output = age_session.run([age_output_name], {age_input_name: image})[0][0]
        age = int(round(float(output[2]) * 100))
        return age if 0 <= age <= 100 else None
    except Exception as e:
        print("Age estimation error:", e)
        return None


# =====================================================
# DASHBOARD UI
# =====================================================
DASHBOARD_W = 1500
DASHBOARD_H = 900
BG = (13, 25, 43)
PANEL = (25, 43, 67)
PANEL2 = (31, 51, 78)
WHITE = (235, 242, 248)
MUTED = (150, 166, 185)
CYAN = (20, 205, 240)
GREEN = (40, 220, 80)
YELLOW = (20, 205, 245)
RED = (50, 80, 240)
PURPLE = (190, 70, 230)
BLUE = (80, 130, 240)
DARK = (20, 31, 48)


def fit_image(image, target_w, target_h):
    """Resize camera image to fill a dashboard panel while preserving aspect ratio."""
    if image is None or image.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h, w = image.shape[:2]
    scale = max(target_w / w, target_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = max(0, (nw - target_w) // 2)
    y = max(0, (nh - target_h) // 2)
    return resized[y:y + target_h, x:x + target_w]


def panel(canvas, x1, y1, x2, y2, title, accent):
    cv2.rectangle(canvas, (x1, y1), (x2, y2), PANEL, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (65, 91, 120), 1)
    cv2.rectangle(canvas, (x1, y1), (x1 + 7, y2), accent, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y1 + 58), PANEL2, -1)
    cv2.line(canvas, (x1, y1 + 58), (x2, y1 + 58), (70, 94, 120), 1)
    cv2.circle(canvas, (x1 + 31, y1 + 29), 7, accent, -1)
    cv2.putText(canvas, title, (x1 + 52, y1 + 39),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, WHITE, 2, cv2.LINE_AA)


def text(canvas, value, x, y, size=0.55, color=WHITE, thickness=1):
    cv2.putText(canvas, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, thickness, cv2.LINE_AA)


def status_badge(canvas, x, y, label, color, width=180):
    cv2.rectangle(canvas, (x, y), (x + width, y + 38), DARK, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + 38), color, 1)
    cv2.circle(canvas, (x + 18, y + 19), 6, color, -1)
    text(canvas, label, x + 32, y + 25, 0.46, color, 1)


def draw_dashboard(camera_frame):
    """Build the dashboard shown in the project presentation."""
    canvas = np.full((DASHBOARD_H, DASHBOARD_W, 3), BG, dtype=np.uint8)

    # Header
    cv2.rectangle(canvas, (24, 18), (1476, 92), PANEL, -1)
    cv2.rectangle(canvas, (24, 18), (1476, 92), (65, 91, 120), 2)
    cv2.circle(canvas, (60, 55), 22, CYAN, 3)
    cv2.line(canvas, (49, 55), (57, 64), GREEN, 4)
    cv2.line(canvas, (57, 64), (72, 44), GREEN, 4)
    text(canvas, "AI BASED DRIVER RECOGNITION & AUTHENTICATION SYSTEM", 94, 52, 0.82, WHITE, 2)
    text(canvas, "EDGE AI  •  VEHICLE SECURITY  •  REAL-TIME AUTHENTICATION", 96, 77, 0.43, CYAN, 1)
    status_badge(canvas, 1250, 35, "SYSTEM READY", GREEN, 220)

    # Panel coordinates
    left1, left2 = 24, 505
    mid1, mid2 = 518, 998
    right1, right2 = 1011, 1490
    top1, top2 = 112, 538
    bot1, bot2 = 550, 860

    # Live feed
    panel(canvas, left1, top1, left2, top2, "LIVE FEED", CYAN)
    cam_x1, cam_y1, cam_x2, cam_y2 = 39, 191, 490, 478
    cam = fit_image(camera_frame, cam_x2-cam_x1, cam_y2-cam_y1)
    canvas[cam_y1:cam_y2, cam_x1:cam_x2] = cam
    cv2.rectangle(canvas, (cam_x1, cam_y1), (cam_x2, cam_y2), YELLOW, 3)
    status_badge(canvas, cam_x1 + 10, cam_y1 + 10, "LIVE", GREEN, 92)
    text(canvas, "READY FOR VERIFICATION" if state == WAITING else
         ("VERIFYING DRIVER..." if state == VERIFYING else
          ("ACCESS GRANTED" if result == "ALLOW" else "ACCESS DENIED")),
         44, 514, 0.52,
         GREEN if result == "ALLOW" else (RED if result == "DENY" else WHITE), 1)

    # Driver details
    panel(canvas, mid1, top1, mid2, top2, "DRIVER DETAILS", PURPLE)
    labels = ["Name", "Driver ID", "Age", "Status", "Liveness", "Confidence", "Face Match"]
    values = [
        driver_name if state != WAITING else "WAITING",
        driver_id if driver_id is not None else "--",
        estimated_age if estimated_age is not None else "--",
        "VERIFIED" if result == "ALLOW" else ("NOT VERIFIED" if result == "DENY" else "NOT VERIFIED"),
        liveness,
        f"{driver_confidence * 100:.2f} %",
        f"{driver_confidence * 100:.2f} %"
    ]
    for i, (lab, val) in enumerate(zip(labels, values)):
        yy = 242 + i * 43
        text(canvas, lab, mid1 + 32, yy, 0.53, MUTED, 1)
        value_color = GREEN if lab == "Status" and result == "ALLOW" else (
            RED if lab == "Status" and result == "DENY" else WHITE)
        text(canvas, val, mid1 + 168, yy, 0.55 if lab != "Status" else 0.58, value_color, 2 if lab == "Status" else 1)

    # System status
    panel(canvas, right1, top1, right2, top2, "SYSTEM STATUS", GREEN)
    cx, cy = 1250, 315
    cv2.circle(canvas, (cx, cy), 62, CYAN, 3)
    cv2.circle(canvas, (cx, cy), 42, (60, 77, 100), 1)
    cv2.circle(canvas, (cx, cy), 14, CYAN, -1)
    text(canvas, "READY" if state == WAITING else ("VERIFYING" if state == VERIFYING else "RESULT"),
         1198, 414, 0.82, CYAN, 2)
    text(canvas, "Vehicle Access", 1080, 463, 0.53, MUTED, 1)
    unlocked = result == "ALLOW"
    lock_color = GREEN if unlocked else CYAN
    # simple lock icon
    lx, ly = 1098, 500
    cv2.rectangle(canvas, (lx, ly + 16), (lx + 30, ly + 46), lock_color, 3)
    cv2.ellipse(canvas, (lx + 15, ly + 16), (10, 13), 180, 0, 180, lock_color, 3)
    text(canvas, "UNLOCKED" if unlocked else "LOCKED", 1145, ly + 39, 0.68, lock_color, 2)

    # Vehicle & location
    panel(canvas, left1, bot1, left2, bot2, "VEHICLE & LOCATION", CYAN)
    text(canvas, "Vehicle ID", 55, 610, 0.53, MUTED, 1)
    text(canvas, "--", 180, 610, 0.55, WHITE, 1)
    text(canvas, "GPS Latitude", 55, 654, 0.53, MUTED, 1)
    text(canvas, "--", 180, 654, 0.55, WHITE, 1)
    text(canvas, "GPS Longitude", 55, 698, 0.53, MUTED, 1)
    text(canvas, "--", 180, 698, 0.55, WHITE, 1)
    status_badge(canvas, 55, 748, "ESP32 TEST MODE" if TEST_MODE else "ESP32 CONNECTED",
                 YELLOW if TEST_MODE else GREEN, 176)
    text(canvas, "ACCESS: " + ("GRANTED" if result == "ALLOW" else "PENDING"),
         260, 773, 0.5, GREEN if result == "ALLOW" else YELLOW, 1)

    # Alerts
    panel(canvas, mid1, bot1, mid2, bot2, "ALERTS", YELLOW)
    alert = "No active alerts"
    alert_color = GREEN
    if result == "DENY":
        alert = "Unauthorized access attempt"
        alert_color = RED
    elif liveness == "SPOOF":
        alert = "Spoofing attempt detected"
        alert_color = RED
    elif image_quality not in ("GOOD", "WAITING", "CHECKING"):
        alert = image_quality
        alert_color = YELLOW
    cv2.rectangle(canvas, (mid1 + 30, bot1 + 88), (mid2 - 30, bot1 + 136), DARK, -1)
    cv2.circle(canvas, (mid1 + 52, bot1 + 112), 7, alert_color, -1)
    text(canvas, alert, mid1 + 72, bot1 + 119, 0.48, alert_color, 1)
    text(canvas, "S", mid1 + 30, bot1 + 206, 0.6, CYAN, 2)
    text(canvas, "START VERIFICATION", mid1 + 58, bot1 + 206, 0.5, MUTED, 1)
    text(canvas, "X", mid1 + 30, bot1 + 244, 0.6, YELLOW, 2)
    text(canvas, "STOP", mid1 + 58, bot1 + 244, 0.5, MUTED, 1)
    text(canvas, "Q", mid1 + 30, bot1 + 282, 0.6, RED, 2)
    text(canvas, "QUIT", mid1 + 58, bot1 + 282, 0.5, MUTED, 1)

    # Date & time
    panel(canvas, right1, bot1, right2, bot2, "DATE & TIME", BLUE)
    now = datetime.now()
    text(canvas, now.strftime("%d %b %Y").upper(), 1165, 685, 0.67, WHITE, 2)
    text(canvas, now.strftime("%I:%M:%S %p"), 1140, 745, 0.86, WHITE, 2)
    text(canvas, "LOCAL SYSTEM TIME", 1180, 780, 0.43, MUTED, 1)
    status_badge(canvas, 1050, 810, "AI ONLINE", GREEN, 145)
    status_badge(canvas, 1208, 810, "EDGE AI", CYAN, 145)

    # Footer
    cv2.line(canvas, (24, 878), (1476, 878), (60, 80, 105), 1)
    text(canvas, "AI SECURITY PLATFORM  •  FACE ID  •  LIVENESS  •  AGE ESTIMATION  •  VEHICLE ACCESS",
         30, 895, 0.35, MUTED, 1)
    text(canvas, "READY", 1405, 895, 0.45, GREEN, 2)

    return canvas


# =====================================================
# MAIN LOOP
# =====================================================
print("\n==============================================")
print(" DRIVER VERIFICATION SYSTEM")
print("==============================================")
print("TEST MODE:", TEST_MODE)
print("S = START | X = STOP | Q = QUIT")
print("==============================================\n")

try:
    while True:
        ok, frame = camera.read()
        if not ok:
            print("Camera frame failed")
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        signal = None

        if not TEST_MODE:
            signal = esp32.read_signal()

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if TEST_MODE:
            if key == ord("s"):
                signal = "START"
                print("TEST ESP32 -> START")
            elif key == ord("x"):
                signal = "STOP"
                print("TEST ESP32 -> STOP")

        # -------------------------------------------------
        # START
        # -------------------------------------------------
        if signal == "START" and state == WAITING:
            print("START received")
            state = VERIFYING
            verification_start_time = time.time()
            result = None
            session_id = datetime.now().strftime("SESSION_%Y%m%d_%H%M%S_%f")
            reset_verification()

        # -------------------------------------------------
        # STOP
        # -------------------------------------------------
        if signal == "STOP":
            print("STOP received")
            state = WAITING
            verification_start_time = None
            result_start_time = None
            result = None
            session_id = None
            reset_verification()

        # -------------------------------------------------
        # WAITING
        # -------------------------------------------------
        if state == WAITING:
            pass

        # -------------------------------------------------
        # VERIFYING
        # -------------------------------------------------
        elif state == VERIFYING:
            face_detector.setInputSize((width, height))
            _, detected_faces = face_detector.detect(frame)
            faces = list(detected_faces) if detected_faces is not None else []
            face_count = len(faces)
            detected_face = face_count > 0
            face = None

            if face_count == 1:
                face = faces[0]
                x, y, w, h = face[:4].astype(int)
                last_bounding_box = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
            elif face_count > 1:
                last_bounding_box = None
                for f in faces:
                    x, y, w, h = f[:4].astype(int)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
            else:
                last_bounding_box = None

            quality_ok, image_quality = check_image_quality(frame, faces)

            if not quality_ok:
                liveness = "WAITING"
                liveness_confidence = 0.0
                driver_id = None
                driver_name = "UNKNOWN"
                driver_confidence = 0.0
                driver_matched = False
                estimated_age = None
                live_counter = 0
                spoof_counter = 0
            else:
                current_live, current_live_conf = check_liveness(frame, face)
                liveness_confidence = current_live_conf

                if current_live == "LIVE":
                    live_counter += 1
                    spoof_counter = 0
                elif current_live == "SPOOF":
                    spoof_counter += 1
                    live_counter = 0
                else:
                    live_counter = 0
                    spoof_counter = 0

                if live_counter >= LIVE_REQUIRED_FRAMES:
                    liveness = "LIVE"
                elif spoof_counter >= SPOOF_REQUIRED_FRAMES:
                    liveness = "SPOOF"
                else:
                    liveness = "CHECKING"

                if liveness == "LIVE":
                    try:
                        recognition = driver_verifier.verify(frame)
                        current_id = recognition.get("driver_id")
                        current_name = recognition.get("driver_name", "UNKNOWN")
                        current_conf = float(recognition.get("confidence", 0.0))
                        matched = current_name != "UNKNOWN" and current_conf >= RECOGNITION_THRESHOLD

                        if matched:
                            driver_id = current_id
                            driver_name = current_name
                            driver_confidence = current_conf
                            driver_matched = True

                            if current_conf > best_driver_confidence:
                                best_driver_id = current_id
                                best_driver_name = current_name
                                best_driver_confidence = current_conf

                            x, y, w, h = face[:4].astype(int)
                            face_crop = crop_face(frame, (x,y,w,h), scale=1.5)
                            age = estimate_age(face_crop)
                            if age is not None:
                                estimated_age = age
                                best_age = age
                        else:
                            driver_id = None
                            driver_name = "UNKNOWN"
                            driver_confidence = max(0.0, current_conf)
                            driver_matched = False
                    except Exception as e:
                        print("Recognition error:", e)
                        driver_id = None
                        driver_name = "UNKNOWN"
                        driver_confidence = 0.0
                        driver_matched = False

                elif liveness == "SPOOF":
                    driver_id = None
                    driver_name = "UNKNOWN"
                    driver_confidence = 0.0
                    driver_matched = False
                    estimated_age = None

            if time.time() - verification_start_time >= PROCESSING_TIME:
                # Use best stable recognition result gathered during the session.
                if best_driver_name != "UNKNOWN" and best_driver_confidence >= RECOGNITION_THRESHOLD:
                    driver_id = best_driver_id
                    driver_name = best_driver_name
                    driver_confidence = best_driver_confidence
                    driver_matched = True
                    if best_age is not None:
                        estimated_age = best_age
                else:
                    driver_id = None
                    driver_name = "UNKNOWN"
                    driver_matched = False

                if image_quality == "GOOD" and liveness == "LIVE" and driver_matched:
                    result = "ALLOW"
                else:
                    result = "DENY"

                vehicle_status = "UNLOCKED" if result == "ALLOW" else "LOCKED"
                processing_time = time.time() - verification_start_time

                verification_data = {
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "session_id": session_id,
                    "face_detection": {
                        "detected": bool(detected_face),
                        "face_count": int(face_count)
                    },
                    "image_quality": {"status": image_quality},
                    "liveness": {
                        "status": liveness,
                        "confidence": float(liveness_confidence)
                    },
                    "driver_authentication": {
                        "driver_id": driver_id,
                        "driver_name": driver_name,
                        "confidence": float(driver_confidence),
                        "matched": bool(driver_matched)
                    },
                    "estimated_age": estimated_age,
                    "decision": {
                        "result": result,
                        "vehicle_status": vehicle_status
                    },
                    "processing_time": float(processing_time)
                }

                print("\n==============================================")
                print("VERIFICATION COMPLETE")
                print("DRIVER:", driver_name)
                print("CONFIDENCE:", f"{driver_confidence:.3f}")
                print("AGE:", estimated_age)
                print("LIVENESS:", liveness, f"({liveness_confidence:.3f})")
                print("IMAGE QUALITY:", image_quality)
                print("RESULT:", result)
                print("VEHICLE:", vehicle_status)
                print("PROCESSING TIME:", f"{processing_time:.2f}s")
                print("==============================================")
                print(json.dumps(verification_data, indent=2))

                try:
                    response = send_verification_result(verification_data)
                    print("API RESPONSE:", response)
                except Exception as e:
                    print("API ERROR:", e)

                if not TEST_MODE:
                    esp32.send_command(result)

                state = RESULT
                result_start_time = time.time()

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------
        elif state == RESULT:
            if time.time() - result_start_time >= RESULT_DISPLAY_TIME:
                state = WAITING
                result = None
                result_start_time = None
                verification_start_time = None
                session_id = None
                reset_verification()

        dashboard = draw_dashboard(frame)
        cv2.imshow(WINDOW_NAME, dashboard)

finally:
    print("Closing application...")
    camera.release()
    cv2.destroyAllWindows()
    if not TEST_MODE:
        esp32.close()
    print("Application closed.")
