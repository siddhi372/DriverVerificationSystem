from client.face_detection import FaceDetector


class AuthenticationPipeline:

    def __init__(self):
        self.face_detector = FaceDetector()

    def process(self, frame):
        faces = self.face_detector.detect(frame)
        return faces