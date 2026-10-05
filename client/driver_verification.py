import json
import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class DriverVerifier:

    def __init__(
        self,
        driver_folder="data/drivers",
        threshold=0.45
    ):

        self.driver_folder = driver_folder
        self.threshold = threshold

        # -----------------------------------------
        # Load InsightFace
        # -----------------------------------------

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        # -----------------------------------------
        # Load registered drivers
        # -----------------------------------------

        self.drivers = self.load_drivers()


    # =============================================
    # LOAD DRIVER DATABASE
    # =============================================

    def load_drivers(self):

        drivers = []

        if not os.path.exists(self.driver_folder):

            print(
                "Driver folder not found:",
                self.driver_folder
            )

            return drivers

        for filename in os.listdir(
            self.driver_folder
        ):

            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(
                self.driver_folder,
                filename
            )

            try:

                with open(
                    file_path,
                    "r"
                ) as file:

                    driver = json.load(file)

                drivers.append(driver)

                print(
                    "Loaded driver:",
                    driver.get("driver_name")
                )

            except Exception as e:

                print(
                    "Could not load:",
                    filename,
                    e
                )

        print(
            "Total registered drivers:",
            len(drivers)
        )

        return drivers


    # =============================================
    # COSINE SIMILARITY
    # =============================================

    def cosine_similarity(
        self,
        embedding1,
        embedding2
    ):

        embedding1 = np.array(
            embedding1,
            dtype=np.float32
        )

        embedding2 = np.array(
            embedding2,
            dtype=np.float32
        )

        # Normalize
        norm1 = np.linalg.norm(
            embedding1
        )

        norm2 = np.linalg.norm(
            embedding2
        )

        if norm1 == 0 or norm2 == 0:

            return 0.0

        embedding1 = (
            embedding1 / norm1
        )

        embedding2 = (
            embedding2 / norm2
        )

        similarity = np.dot(
            embedding1,
            embedding2
        )

        return float(similarity)


    # =============================================
    # VERIFY FACE
    # =============================================

    def verify(self, frame):

        # -----------------------------------------
        # Detect faces
        # -----------------------------------------

        faces = self.app.get(frame)


        # No face
        if len(faces) == 0:

            return {
                "verified": False,
                "driver_name": "UNKNOWN",
                "driver_id": None,
                "confidence": 0.0
            }


        # Multiple faces
        if len(faces) > 1:

            return {
                "verified": False,
                "driver_name": "UNKNOWN",
                "driver_id": None,
                "confidence": 0.0
            }


        # -----------------------------------------
        # Get face
        # -----------------------------------------

        face = faces[0]


        # -----------------------------------------
        # Get live embedding
        # -----------------------------------------

        live_embedding = face.embedding


        # Normalize
        live_embedding = (
            live_embedding /
            np.linalg.norm(live_embedding)
        )


        # -----------------------------------------
        # Compare with registered drivers
        # -----------------------------------------

        best_driver = None
        best_similarity = -1.0


        for driver in self.drivers:

            stored_embedding = driver.get(
                "embedding"
            )

            if stored_embedding is None:
                continue


            similarity = self.cosine_similarity(
                live_embedding,
                stored_embedding
            )


            if similarity > best_similarity:

                best_similarity = similarity

                best_driver = driver


        # -----------------------------------------
        # Check threshold
        # -----------------------------------------

        if (
            best_driver is not None
            and
            best_similarity >= self.threshold
        ):

            return {
                "verified": True,

                "driver_name":
                    best_driver.get(
                        "driver_name",
                        "UNKNOWN"
                    ),

                "driver_id":
                    best_driver.get(
                        "driver_id"
                    ),

                "confidence":
                    best_similarity
            }


        # -----------------------------------------
        # Unknown driver
        # -----------------------------------------

        return {
            "verified": False,

            "driver_name":
                "UNKNOWN",

            "driver_id":
                None,

            "confidence":
                max(
                    best_similarity,
                    0.0
                )
        }