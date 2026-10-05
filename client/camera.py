import cv2
import time
from pathlib import Path

from client.authentication import AuthenticationPipeline



# -----------------------------
# Camera Configuration
# -----------------------------

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

CAPTURE_FOLDER = Path("client/captured_frames")
CAPTURE_FOLDER.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Open Camera
# -----------------------------

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    print("Could not access camera.")
    exit()


# -----------------------------
# Authentication Pipeline
# -----------------------------

authentication = AuthenticationPipeline()


# -----------------------------
# UI
# -----------------------------

window_name = "Driver Security - Authentication"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1100, 700)


# -----------------------------
# Main Camera Loop
# -----------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # Send frame to AI pipeline
    faces = authentication.process(frame)

    # Draw detected faces
    if faces is not None:

        for face in faces:

            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    # -----------------------------
    # Display Status
    # -----------------------------

    cv2.putText(
        frame,
        "DRIVER AUTHENTICATION",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press C to capture frame",
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (30, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Show live webcam
    cv2.imshow(window_name, frame)

    # Read keyboard
    key = cv2.waitKey(1) & 0xFF

    # Capture frame
    if key == ord("c"):

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        filename = CAPTURE_FOLDER / f"auth_{timestamp}.jpg"

        cv2.imwrite(str(filename), frame)

        print(f"Authentication frame captured: {filename}")

    # Quit
    elif key == ord("q"):
        break


# -----------------------------
# Release Resources
# -----------------------------

camera.release()
cv2.destroyAllWindows()