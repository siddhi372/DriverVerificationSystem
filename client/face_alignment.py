import cv2
import numpy as np


# ---------------------------------
# YuNet Model
# ---------------------------------

MODEL_PATH = "models/face_detection_yunet_2026may.onnx"

detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)


# ---------------------------------
# Open Camera
# ---------------------------------

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Could not access camera")
    exit()


# ---------------------------------
# Main Loop
# ---------------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read frame")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    # Set YuNet input size
    detector.setInputSize((width, height))

    # Detect faces
    _, faces = detector.detect(frame)


    # ---------------------------------
    # Process Faces
    # ---------------------------------

    if faces is not None:

        for face in faces:

            # ---------------------------------
            # Get the two eye landmarks
            # ---------------------------------

            eye1 = np.array(
                [face[4], face[5]],
                dtype=np.float32
            )

            eye2 = np.array(
                [face[6], face[7]],
                dtype=np.float32
            )


            # ---------------------------------
            # Determine spatial left/right eyes
            #
            # We use X position rather than
            # YuNet's anatomical labels because
            # the webcam image is mirrored.
            # ---------------------------------

            if eye1[0] < eye2[0]:

                left_eye = eye1
                right_eye = eye2

            else:

                left_eye = eye2
                right_eye = eye1


            # ---------------------------------
            # Calculate eye angle
            # ---------------------------------

            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]

            angle = np.degrees(
                np.arctan2(dy, dx)
            )


            # ---------------------------------
            # Face Center
            # ---------------------------------

            center = (
                int((left_eye[0] + right_eye[0]) / 2),
                int((left_eye[1] + right_eye[1]) / 2)
            )


            # ---------------------------------
            # Rotate Face
            # ---------------------------------

            rotation_matrix = cv2.getRotationMatrix2D(
                center,
                angle,
                1.0
            )

            rotated = cv2.warpAffine(
                frame,
                rotation_matrix,
                (width, height)
            )


            # ---------------------------------
            # Original Face Bounding Box
            # ---------------------------------

            x, y, w, h = face[:4].astype(int)


            # ---------------------------------
            # Transform Bounding Box Corners
            # ---------------------------------

            corners = np.array([
                [x, y],
                [x + w, y],
                [x, y + h],
                [x + w, y + h]
            ], dtype=np.float32)

            corners_with_ones = np.hstack([
                corners,
                np.ones((4, 1), dtype=np.float32)
            ])

            transformed = (
                rotation_matrix @ corners_with_ones.T
            ).T


            new_x1 = int(transformed[:, 0].min())
            new_y1 = int(transformed[:, 1].min())

            new_x2 = int(transformed[:, 0].max())
            new_y2 = int(transformed[:, 1].max())


            # ---------------------------------
            # Padding
            # ---------------------------------

            padding = 30

            new_x1 = max(0, new_x1 - padding)
            new_y1 = max(0, new_y1 - padding)

            new_x2 = min(width, new_x2 + padding)
            new_y2 = min(height, new_y2 + padding)


            # ---------------------------------
            # Crop
            # ---------------------------------

            aligned_face = rotated[
                new_y1:new_y2,
                new_x1:new_x2
            ]


            if aligned_face.size == 0:
                continue


            # ---------------------------------
            # Resize to Model Input
            # ---------------------------------

            aligned_face = cv2.resize(
                aligned_face,
                (112, 112)
            )


            # ---------------------------------
            # Display Larger Preview
            # ---------------------------------

            display_face = cv2.resize(
                aligned_face,
                (336, 336)
            )

            cv2.imshow(
                "Aligned Face - 112x112",
                display_face
            )


            # ---------------------------------
            # Draw Original Face Box
            # ---------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )


            # ---------------------------------
            # Draw YuNet Landmarks
            # ---------------------------------

            landmarks = [
                [face[4], face[5]],
                [face[6], face[7]],
                [face[8], face[9]],
                [face[10], face[11]],
                [face[12], face[13]]
            ]

            for point in landmarks:

                px = int(point[0])
                py = int(point[1])

                cv2.circle(
                    frame,
                    (px, py),
                    4,
                    (0, 0, 255),
                    -1
                )


            # ---------------------------------
            # Status
            # ---------------------------------

            cv2.putText(
                frame,
                "FACE ALIGNED",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


    # ---------------------------------
    # Original Camera Window
    # ---------------------------------

    cv2.putText(
        frame,
        "Press Q to quit",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Original Camera",
        frame
    )


    # ---------------------------------
    # Quit
    # ---------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ---------------------------------
# Release
# ---------------------------------

camera.release()
cv2.destroyAllWindows()