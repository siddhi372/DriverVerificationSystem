from datetime import datetime
import uuid


def create_verification_result(
    face_detected,
    face_count,
    bounding_box,
    image_quality,
    liveness,
    liveness_confidence,
    driver_name,
    driver_confidence,
    decision
):
    """
    Combine all AI verification results into
    one JSON-compatible Python dictionary.
    """

    result = {

        # ---------------------------------------------
        # Verification session
        # ---------------------------------------------

        "session_id": (
            "VER_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + "_"
            + uuid.uuid4().hex[:6]
        ),

        "timestamp": datetime.now().isoformat(),

        # ---------------------------------------------
        # Face Detection
        # ---------------------------------------------

        "face_detection": {

            "detected": bool(
                face_detected
            ),

            "face_count": int(
                face_count
            ),

            "bounding_box": bounding_box
        },

        # ---------------------------------------------
        # Image Quality
        # ---------------------------------------------

        "image_quality": {

            "status": image_quality
        },

        # ---------------------------------------------
        # Liveness
        # ---------------------------------------------

        "liveness": {

            "status": liveness,

            "confidence": float(
                liveness_confidence
            )
        },

        # ---------------------------------------------
        # Driver Recognition
        # ---------------------------------------------

        "driver_authentication": {

            "driver_name": driver_name,

            "confidence": float(
                driver_confidence
            ),

            "matched": (
                driver_name != "UNKNOWN"
            )
        },

        # ---------------------------------------------
        # Final Decision
        # ---------------------------------------------

        "decision": {

            "result": decision,

            "vehicle_status": (
                "UNLOCKED"
                if decision == "ALLOW"
                else "LOCKED"
            )
        }
    }

    return result