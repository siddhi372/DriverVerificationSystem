import cv2

MODEL_PATH = "models/face_detection_yunet_2026may.onnx"


class FaceDetector:

    def __init__(self):

        self.detector = cv2.FaceDetectorYN.create(
            MODEL_PATH,
            "",
            (320, 320),
            0.6,
            0.3,
            5000
        )

    def detect(self, frame):

        height, width = frame.shape[:2]

        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(frame)

        return faces