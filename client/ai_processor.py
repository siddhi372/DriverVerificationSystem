"""
AI PROCESSOR
------------
All AI/image-processing logic for the Driver Verification System.

UI code should NOT perform face detection, liveness, recognition or age
estimation directly.  The UI should call:

    processor.process(frame)

and display the returned dictionary.
"""

import os
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


class AIProcessor:
    # Models / thresholds
    FACE_MODEL = "models/face_detection_yunet_2026may.onnx"
    LIVENESS_MODEL = "models/MiniFASNetV2.onnx"
    DRIVER_FOLDER = "data/drivers"

    RECOGNITION_THRESHOLD = 0.40
    LIVE_REQUIRED_FRAMES = 5
    SPOOF_REQUIRED_FRAMES = 3

    def __init__(self):
        print("\nInitializing AI Processor...")

        # -----------------------------
        # YuNet face detector
        # -----------------------------
        self.detector = cv2.FaceDetectorYN.create(
            self.FACE_MODEL,
            "",
            (320, 320),
            0.6,
            0.3,
            5000
        )

        # -----------------------------
        # MiniFASNet liveness
        # -----------------------------
        self.liveness_session = ort.InferenceSession(
            self.LIVENESS_MODEL,
            providers=["CPUExecutionProvider"]
        )
        self.liveness_input = self.liveness_session.get_inputs()[0].name
        self.liveness_output = self.liveness_session.get_outputs()[0].name

        # -----------------------------
        # ArcFace / InsightFace
        # -----------------------------
        self.recognition_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.recognition_app.prepare(
            ctx_id=0,
            det_size=(320, 320)
        )

        # -----------------------------
        # Age model
        # -----------------------------
        self.age_session = self._load_age_model()
        self.age_input = None
        self.age_output = None

        if self.age_session is not None:
            self.age_input = self.age_session.get_inputs()[0].name
            self.age_output = self.age_session.get_outputs()[0].name

        # -----------------------------
        # Registered driver embeddings
        # -----------------------------
        self.registered_drivers = self._load_drivers()

        # Liveness stability
        self.live_count = 0
        self.spoof_count = 0

        print(
            f"AI Processor ready | "
            f"Registered drivers: {len(self.registered_drivers)}"
        )

    # =========================================================
    # MODEL LOADING
    # =========================================================

    def _load_age_model(self):
        """
        Load the InsightFace gender/age model.

        First checks the original Windows location used by the project.
        Then checks the normal InsightFace buffalo_l directory.
        """
        candidates = []

        original_path = r"C:\Users\Siddhi\.insightface\models\buffalo_l\genderage.onnx"
        candidates.append(original_path)

        home = Path.home()
        candidates.append(
            str(home / ".insightface" / "models" / "buffalo_l" / "genderage.onnx")
        )

        for path in candidates:
            if os.path.exists(path):
                try:
                    session = ort.InferenceSession(
                        path,
                        providers=["CPUExecutionProvider"]
                    )
                    print(f"Age model loaded: {path}")
                    return session
                except Exception as exc:
                    print(f"Age model could not be loaded: {exc}")

        print("WARNING: Age model not found. Age will be None.")
        return None

    def _load_drivers(self):
        drivers = []

        folder = Path(self.DRIVER_FOLDER)

        if not folder.exists():
            print(f"WARNING: Driver folder not found: {folder}")
            return drivers

        for file_path in folder.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                embedding = np.asarray(
                    data["embedding"],
                    dtype=np.float32
                )

                norm = np.linalg.norm(embedding)

                if norm == 0:
                    continue

                embedding = embedding / norm

                drivers.append({
                    "driver_id": data["driver_id"],
                    "driver_name": data["driver_name"],
                    "embedding": embedding
                })

                print(
                    f"Loaded driver: "
                    f"{data['driver_id']} - {data['driver_name']}"
                )

            except Exception as exc:
                print(f"Could not load {file_path.name}: {exc}")

        return drivers

    # =========================================================
    # STATE
    # =========================================================

    def reset(self):
        """Reset temporary liveness state."""
        self.live_count = 0
        self.spoof_count = 0

    # =========================================================
    # FACE QUALITY
    # =========================================================

    def check_face_quality(self, frame, face, landmarks):
        height, width = frame.shape[:2]

        x, y, w, h = face[:4].astype(int)

        # Face must be completely inside the camera frame.
        if x < 0 or y < 0 or x + w > width or y + h > height:
            return False, "SHOW YOUR FULL FACE"

        # Face must be large enough.
        if w < 150 or h < 150:
            return False, "COME CLOSER"

        face_crop = frame[y:y + h, x:x + w]

        if face_crop.size == 0:
            return False, "SHOW YOUR FACE"

        # Lighting check.
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))

        if brightness < 45:
            return False, "LIGHT TOO DIM"

        # Look-straight check using YuNet landmarks.
        left_eye = np.asarray([landmarks[0], landmarks[1]], dtype=np.float32)
        nose = np.asarray(landmarks[2], dtype=np.float32)

        eye_center = np.mean(left_eye, axis=0)
        eye_distance = abs(
            float(landmarks[1][0]) - float(landmarks[0][0])
        )

        if eye_distance > 0:
            nose_offset = abs(
                float(nose[0]) - float(eye_center[0])
            )
            ratio = nose_offset / eye_distance

            if ratio > 0.45:
                return False, "LOOK STRAIGHT"

        return True, "GOOD"

    # =========================================================
    # FACE ALIGNMENT
    # =========================================================

    @staticmethod
    def align_face(frame, landmarks):
        target = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

        landmarks = np.asarray(landmarks, dtype=np.float32)

        matrix, _ = cv2.estimateAffinePartial2D(
            landmarks,
            target,
            method=cv2.LMEDS
        )

        if matrix is None:
            return None

        return cv2.warpAffine(
            frame,
            matrix,
            (112, 112)
        )

    # =========================================================
    # LIVENESS
    # =========================================================

    def check_liveness(self, aligned_face):
        face = cv2.resize(aligned_face, (80, 80))
        face = face.astype(np.float32)
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)

        output = self.liveness_session.run(
            [self.liveness_output],
            {self.liveness_input: face}
        )[0]

        # Stable softmax
        probabilities = np.exp(
            output - np.max(output, axis=1, keepdims=True)
        )
        probabilities /= probabilities.sum(
            axis=1,
            keepdims=True
        )

        probabilities = probabilities[0]

        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])

        # Existing project convention:
        # class 1 = LIVE, class 0/2 = SPOOF.
        label = "LIVE" if prediction == 1 else "SPOOF"

        return label, confidence

    def stable_liveness(self, label):
        if label == "LIVE":
            self.live_count += 1
            self.spoof_count = 0
        else:
            self.spoof_count += 1
            self.live_count = 0

        if self.live_count >= self.LIVE_REQUIRED_FRAMES:
            return "LIVE"

        if self.spoof_count >= self.SPOOF_REQUIRED_FRAMES:
            return "SPOOF"

        return "CHECKING"

    # =========================================================
    # FACE RECOGNITION
    # =========================================================

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return -1.0

        embedding1 = embedding1 / norm1
        embedding2 = embedding2 / norm2

        return float(np.dot(embedding1, embedding2))

    def verify_driver(self, aligned_face):
        faces = self.recognition_app.get(aligned_face)

        if len(faces) == 0:
            return "UNKNOWN", 0.0, None

        current_embedding = faces[0].embedding

        if current_embedding is None:
            return "UNKNOWN", 0.0, None

        current_embedding = np.asarray(
            current_embedding,
            dtype=np.float32
        )

        norm = np.linalg.norm(current_embedding)

        if norm == 0:
            return "UNKNOWN", 0.0, None

        current_embedding /= norm

        best_similarity = -1.0
        best_driver = None

        for driver in self.registered_drivers:
            similarity = self.cosine_similarity(
                driver["embedding"],
                current_embedding
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_driver = driver

        if (
            best_driver is not None
            and best_similarity >= self.RECOGNITION_THRESHOLD
        ):
            return "MATCH", best_similarity, best_driver

        return "UNKNOWN", best_similarity, None

    # =========================================================
    # AGE ESTIMATION
    # =========================================================

    def estimate_age(self, aligned_face):
        if self.age_session is None:
            return None

        face = cv2.resize(aligned_face, (96, 96))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32)
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)

        output = self.age_session.run(
            [self.age_output],
            {self.age_input: face}
        )[0][0]

        # Existing project convention for buffalo_l genderage.onnx.
        age = int(round(float(output[2]) * 100))

        return age

    # =========================================================
    # MAIN PROCESSING FUNCTION
    # =========================================================

    def process(self, frame):
        """
        Process one camera frame.

        Returns a dictionary designed for the UI.
        No OpenCV drawing is performed here.
        """

        result = {
            "face_detected": False,
            "face_count": 0,
            "bounding_box": None,
            "image_quality": "WAITING",
            "guidance": "SHOW YOUR FACE",
            "liveness": "WAITING",
            "liveness_confidence": 0.0,
            "identity": "UNKNOWN",
            "driver_id": None,
            "driver_name": "UNKNOWN",
            "driver_confidence": 0.0,
            "matched": False,
            "estimated_age": None,
            "decision": "DENY",
            "vehicle_status": "LOCKED",
        }

        if frame is None or frame.size == 0:
            self.reset()
            return result

        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(frame)

        if faces is None or len(faces) == 0:
            self.reset()
            return result

        result["face_detected"] = True
        result["face_count"] = len(faces)

        # More than one face is not allowed.
        if len(faces) > 1:
            self.reset()
            result["image_quality"] = "MULTIPLE_FACES"
            result["guidance"] = "ONLY ONE FACE ALLOWED"
            return result

        face = faces[0]

        x, y, w, h = face[:4].astype(int)

        result["bounding_box"] = {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        }

        # YuNet gives five landmark points.
        landmarks = np.array([
            [face[4], face[5]],
            [face[6], face[7]],
            [face[8], face[9]],
            [face[10], face[11]],
            [face[12], face[13]]
        ], dtype=np.float32)

        quality_ok, message = self.check_face_quality(
            frame,
            face,
            landmarks
        )

        if not quality_ok:
            self.reset()
            result["image_quality"] = "POOR"
            result["guidance"] = message
            return result

        result["image_quality"] = "GOOD"
        result["guidance"] = "CHECKING"

        aligned_face = self.align_face(
            frame,
            landmarks
        )

        if aligned_face is None:
            self.reset()
            result["image_quality"] = "POOR"
            result["guidance"] = "ADJUST YOUR FACE"
            return result

        # Liveness is always checked before identity.
        live_label, live_confidence = self.check_liveness(
            aligned_face
        )

        stable_label = self.stable_liveness(live_label)

        result["liveness"] = stable_label
        result["liveness_confidence"] = round(
            live_confidence,
            3
        )

        if stable_label == "CHECKING":
            result["guidance"] = "HOLD STILL"
            return result

        if stable_label == "SPOOF":
            result["guidance"] = "SPOOF DETECTED"
            result["decision"] = "DENY"
            result["vehicle_status"] = "LOCKED"
            return result

        # Stable LIVE -> recognize driver.
        identity, similarity, driver = self.verify_driver(
            aligned_face
        )

        result["identity"] = identity
        result["driver_confidence"] = round(
            max(0.0, similarity),
            3
        )

        if identity == "MATCH" and driver is not None:
            result["driver_id"] = driver["driver_id"]
            result["driver_name"] = driver["driver_name"]
            result["matched"] = True

            # Age is evaluated after identity match.
            result["estimated_age"] = self.estimate_age(
                aligned_face
            )

            result["guidance"] = "VERIFIED"
            result["decision"] = "ALLOW"
            result["vehicle_status"] = "UNLOCKED"

        else:
            result["driver_name"] = "UNKNOWN"
            result["guidance"] = "UNKNOWN DRIVER"
            result["decision"] = "DENY"
            result["vehicle_status"] = "LOCKED"

        return result
