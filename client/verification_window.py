import cv2
import time

from client.esp32_simulator import ESP32Simulator
from client.face_detection import FaceDetector


# =====================================================
# CONFIGURATION
# =====================================================

CAMERA_INDEX = 0

PROCESSING_TIME = 5


# =====================================================
# STATES
# =====================================================

WAITING = "WAITING"
PROCESSING = "PROCESSING"
ALLOWED = "ALLOWED"
DENIED = "DENIED"


# =====================================================
# ESP32 SIMULATOR
# =====================================================

esp32 = ESP32Simulator()


# =====================================================
# AI
# =====================================================

face_detector = FaceDetector()


# =====================================================
# CAMERA
# =====================================================

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

camera.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not camera.isOpened():

    print("Could not access camera.")
    exit()


# =====================================================
# WINDOW
# =====================================================

WINDOW_NAME = "Driver Verification System"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# =====================================================
# INITIAL STATE
# =====================================================

state = WAITING

verification_start_time = None

face_detected = False


# =====================================================
# MAIN LOOP
# =====================================================

while True:

    # -------------------------------------------------
    # Read camera
    # -------------------------------------------------

    success, frame = camera.read()

    if not success:

        print("Camera frame failed")
        break


    # Mirror camera
    frame = cv2.flip(
        frame,
        1
    )


    # -------------------------------------------------
    # Read keyboard
    #
    # S = ESP32 START
    # X = ESP32 STOP
    #
    # This is ONLY for testing.
    # -------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    signal = esp32.read_signal(key)


    if signal is not None:

        print("ESP32 SIMULATOR →", signal)


    # =================================================
    # STOP SIGNAL
    # =================================================

    if signal == "STOP":

        print("STOP received")

        state = WAITING

        face_detected = False

        verification_start_time = None


    # =================================================
    # WAITING STATE
    # =================================================

    if state == WAITING:

        if signal == "START":

            print("START received")

            state = PROCESSING

            verification_start_time = time.time()

            face_detected = False


    # =================================================
    # PROCESSING STATE
    # =================================================

    elif state == PROCESSING:

        # ---------------------------------------------
        # Face detection
        # ---------------------------------------------

        height, width = frame.shape[:2]

        face_detector.detector.setInputSize(
            (width, height)
        )

        _, faces = face_detector.detector.detect(
            frame
        )


        # ---------------------------------------------
        # Check face
        # ---------------------------------------------

        if faces is not None and len(faces) > 0:

            face_detected = True

            face = faces[0]

            x, y, w, h = (
                face[:4].astype(int)
            )


            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

        else:

            face_detected = False


        # ---------------------------------------------
        # Processing time
        # ---------------------------------------------

        elapsed = (
            time.time()
            - verification_start_time
        )


        # ---------------------------------------------
        # UI
        # ---------------------------------------------

        cv2.putText(
            frame,
            "DRIVER VERIFICATION",
            (50, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 255, 0),
            3
        )


        cv2.putText(
            frame,
            "PLEASE WAIT...",
            (50, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )


        if face_detected:

            cv2.putText(
                frame,
                "FACE DETECTED",
                (50, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "LOOK AT THE CAMERA",
                (50, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )


        # ---------------------------------------------
        # TEMPORARY DECISION
        #
        # For testing ONLY:
        # Face detected for 5 seconds = ALLOW
        #
        # Later this will become:
        #
        # Face Detection
        #      ↓
        # Face Alignment
        #      ↓
        # Liveness
        #      ↓
        # Face Recognition
        #      ↓
        # Age Estimation
        #      ↓
        # Server Decision
        #      ↓
        # ALLOW / DENY
        # ---------------------------------------------

        if elapsed >= PROCESSING_TIME:

            if face_detected:

                state = ALLOWED

                esp32.send_command(
                    "ALLOW"
                )

            else:

                state = DENIED

                esp32.send_command(
                    "DENY"
                )


    # =================================================
    # ALLOWED STATE
    # =================================================

    elif state == ALLOWED:

        cv2.putText(
            frame,
            "ACCESS GRANTED",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            4
        )

        cv2.putText(
            frame,
            "VEHICLE UNLOCKED",
            (50, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            "WAITING FOR SWITCH OFF",
            (50, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # =================================================
    # DENIED STATE
    # =================================================

    elif state == DENIED:

        cv2.putText(
            frame,
            "ACCESS DENIED",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            4
        )

        cv2.putText(
            frame,
            "VEHICLE LOCKED",
            (50, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            "WAITING FOR SWITCH OFF",
            (50, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # =================================================
    # WAITING SCREEN
    # =================================================

    if state == WAITING:

        cv2.putText(
            frame,
            "DRIVER VERIFICATION SYSTEM",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame,
            "WAITING FOR VEHICLE SIGNAL...",
            (50, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )


    # =================================================
    # DISPLAY
    # =================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


# =====================================================
# CLEANUP
# =====================================================

camera.release()

cv2.destroyAllWindows()