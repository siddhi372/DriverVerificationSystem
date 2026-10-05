import cv2
import numpy as np


MODEL_PATH = "models/MiniFASNetV2.onnx"


class LivenessDetector:

    def __init__(self):

        # Load MiniFASNetV2
        self.net = cv2.dnn.readNetFromONNX(
            MODEL_PATH
        )

        self.input_size = (80, 80)


    def predict(self, face):

        # ---------------------------------
        # Resize face to 80 x 80
        # ---------------------------------

        face = cv2.resize(
            face,
            self.input_size
        )


        # ---------------------------------
        # Convert image to blob
        # ---------------------------------

        blob = cv2.dnn.blobFromImage(
            face,
            scalefactor=1.0 / 255.0,
            size=self.input_size,
            mean=(0, 0, 0),
            swapRB=False,
            crop=False
        )


        # ---------------------------------
        # Give input to model
        # ---------------------------------

        self.net.setInput(blob)


        # ---------------------------------
        # Get output
        # ---------------------------------

        output = self.net.forward()


        # ---------------------------------
        # Convert output to 1D array
        # ---------------------------------

        scores = output.flatten()


        # ---------------------------------
        # Softmax
        # ---------------------------------

        exp_scores = np.exp(
            scores - np.max(scores)
        )

        probabilities = (
            exp_scores /
            np.sum(exp_scores)
        )


        # ---------------------------------
        # Find highest probability
        # ---------------------------------

        class_id = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[class_id]
        )


        return class_id, confidence, probabilities