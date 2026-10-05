import cv2

from client.driver_verification import DriverVerifier


# =====================================================
# DRIVER VERIFIER
# =====================================================

verifier = DriverVerifier(
    driver_folder="data/drivers",
    threshold=0.45
)


# =====================================================
# CAMERA
# =====================================================

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not camera.isOpened():

    print("Could not access camera.")
    exit()


# =====================================================
# MAIN LOOP
# =====================================================

while True:

    success, frame = camera.read()

    if not success:

        print("Could not read camera frame.")
        break


    # Mirror webcam
    frame = cv2.flip(
        frame,
        1
    )


    # =================================================
    # DRIVER VERIFICATION
    # =================================================

    result = verifier.verify(
        frame
    )


    # =================================================
    # DISPLAY RESULT
    # =================================================

    driver_name = result["driver_name"]

    confidence = result["confidence"]

    verified = result["verified"]


    if verified:

        status = "VERIFIED"

        status_color = (
            0,
            255,
            0
        )

    else:

        status = "UNKNOWN"

        status_color = (
            0,
            0,
            255
        )


    # Driver name

    cv2.putText(
        frame,
        f"DRIVER: {driver_name}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        status_color,
        2
    )


    # Status

    cv2.putText(
        frame,
        f"STATUS: {status}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2
    )


    # Similarity

    cv2.putText(
        frame,
        f"SIMILARITY: {confidence:.3f}",
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # =================================================
    # DISPLAY
    # =================================================

    cv2.imshow(
        "Driver Recognition Test",
        frame
    )


    # =================================================
    # QUIT
    # =================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# =====================================================
# CLEANUP
# =====================================================

camera.release()

cv2.destroyAllWindows()