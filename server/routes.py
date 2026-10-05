import os
import json
import base64
import numpy as np
from datetime import datetime

from fastapi import APIRouter, HTTPException, Form, File, UploadFile

from database.database import (
    save_verification_result,
    get_verification_history,
    get_verification_summary,
    get_all_drivers,
    get_driver,
    add_driver,
    update_driver_status,
    update_driver_vehicle,
    get_connection
)


# =====================================================
# ROUTER
# =====================================================

router = APIRouter()


# =====================================================
# VERIFICATION RESULT API
# =====================================================

@router.post("/verification/result")
def verification_result(data: dict):

    print(
        "\n=============================="
    )

    print(
        "VERIFICATION RESULT RECEIVED"
    )

    print(
        data
    )

    print(
        "=============================="
    )

    # -------------------------------------------------
    # SAVE RESULT
    # -------------------------------------------------

    try:

        record_id = save_verification_result(
            data
        )

    except Exception as e:

        print(
            "DATABASE ERROR:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to save verification result"
        )

    print(
        "DATABASE RECORD ID:",
        record_id
    )

    return {

        "status": "success",

        "message":
            "Verification result received",

        "decision":
            data.get(
                "decision",
                "UNKNOWN"
            ),

        "record_id":
            record_id

    }


# =====================================================
# VERIFICATION HISTORY
# =====================================================

@router.get("/verification/history")
def verification_history():

    try:

        history = get_verification_history(
            50
        )

        return {

            "status":
                "success",

            "history":
                history

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =====================================================
# VERIFICATION SUMMARY
# =====================================================

@router.get("/verification/summary")
def verification_summary():

    try:

        summary = get_verification_summary()

        return {

            "status":
                "success",

            "summary":
                summary

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =====================================================
# DRIVER MANAGEMENT
# =====================================================


# =====================================================
# GET ALL DRIVERS
# =====================================================

@router.get("/drivers")
def get_drivers():

    try:

        drivers = get_all_drivers()

        return {

            "status":
                "success",

            "drivers":
                drivers

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =====================================================
# GET SINGLE DRIVER
# =====================================================

@router.get("/drivers/{driver_id}")
def get_single_driver(
    driver_id: str
):

    driver = get_driver(
        driver_id
    )

    if driver is None:

        raise HTTPException(
            status_code=404,
            detail="Driver not found"
        )

    return {

        "status":
            "success",

        "driver":
            driver

    }


# =====================================================
# ADD DRIVER
# =====================================================

@router.post("/drivers")
def create_driver(
    data: dict
):

    driver_id = data.get(
        "driver_id"
    )

    name = data.get(
        "name"
    )

    assigned_vehicle = data.get(
        "assigned_vehicle"
    )

    if not driver_id:

        raise HTTPException(
            status_code=400,
            detail="driver_id is required"
        )

    if not name:

        raise HTTPException(
            status_code=400,
            detail="name is required"
        )

    try:

        database_id = add_driver(

            driver_id=
                driver_id,

            name=
                name,

            assigned_vehicle=
                assigned_vehicle

        )

        return {

            "status":
                "success",

            "message":
                "Driver created successfully",

            "database_id":
                database_id

        }

    except Exception as e:

        raise HTTPException(

            status_code=400,

            detail=str(e)

        )


# =====================================================
# UPDATE DRIVER STATUS
# =====================================================

@router.put("/drivers/{driver_id}/status")
def change_driver_status(

    driver_id: str,

    data: dict

):

    status = data.get(
        "status"
    )

    if status not in [
        "ACTIVE",
        "INACTIVE"
    ]:

        raise HTTPException(

            status_code=400,

            detail=
                "Status must be ACTIVE or INACTIVE"

        )

    driver = get_driver(
        driver_id
    )

    if driver is None:

        raise HTTPException(

            status_code=404,

            detail=
                "Driver not found"

        )

    update_driver_status(

        driver_id=
            driver_id,

        status=
            status

    )

    return {

        "status":
            "success",

        "message":
            "Driver status updated",

        "driver_id":
            driver_id,

        "new_status":
            status

    }


# =====================================================
# ASSIGN VEHICLE
# =====================================================

@router.put("/drivers/{driver_id}/vehicle")
def change_driver_vehicle(

    driver_id: str,

    data: dict

):

    assigned_vehicle = data.get(
        "assigned_vehicle"
    )

    driver = get_driver(
        driver_id
    )

    if driver is None:

        raise HTTPException(

            status_code=404,

            detail=
                "Driver not found"

        )

    update_driver_vehicle(

        driver_id=
            driver_id,

        assigned_vehicle=
            assigned_vehicle

    )

    return {

        "status":
            "success",

        "message":
            "Driver vehicle updated",

        "driver_id":
            driver_id,

        "assigned_vehicle":
            assigned_vehicle

    }

# =====================================================
# DRIVER ENROLLMENT
# =====================================================

# =====================================================
# DRIVER ENROLLMENT WITH AI FACE EMBEDDING
# =====================================================

from fastapi import UploadFile, File, Form, HTTPException
import json
import os

from server.ai_enrollment import generate_face_embedding


@router.post("/drivers/enroll")
async def enroll_driver(
    driver_id: str = Form(...),
    name: str = Form(...),
    address: str = Form(""),
    age: int = Form(...),
    enrollment_image: UploadFile = File(...)
):

    conn = get_connection()

    try:

        # -------------------------------------------------
        # Validate basic information
        # -------------------------------------------------

        driver_id = driver_id.strip()
        name = name.strip()
        address = address.strip()

        if not driver_id:
            raise HTTPException(
                status_code=400,
                detail="Driver ID is required"
            )

        if not name:
            raise HTTPException(
                status_code=400,
                detail="Driver name is required"
            )

        if age < 18:
            raise HTTPException(
                status_code=400,
                detail="Driver must be at least 18 years old"
            )


        # -------------------------------------------------
        # Check duplicate driver
        # -------------------------------------------------

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT driver_id
                FROM drivers
                WHERE driver_id = %s
                """,
                (driver_id,)
            )

            if cur.fetchone():

                raise HTTPException(
                    status_code=409,
                    detail="Driver ID already exists"
                )


        # -------------------------------------------------
        # Read enrollment image
        # -------------------------------------------------

        image_bytes = await enrollment_image.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Enrollment image is empty"
            )


        # -------------------------------------------------
        # AI FACE EMBEDDING
        # -------------------------------------------------

        try:

            ai_result = generate_face_embedding(
                image_bytes
            )

        except ValueError as e:

            raise HTTPException(
                status_code=400,
                detail=str(e)
            )


        embedding = ai_result["embedding"]


        # -------------------------------------------------
        # Save enrollment image
        # -------------------------------------------------

        os.makedirs(
            "uploads/enrollment",
            exist_ok=True
        )

        image_path = (
            f"uploads/enrollment/"
            f"{driver_id}.jpg"
        )

        with open(
            image_path,
            "wb"
        ) as file:

            file.write(image_bytes)


        # -------------------------------------------------
        # Save driver + embedding
        # -------------------------------------------------

        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO drivers
                (
                    driver_id,
                    name,
                    address,
                    age,
                    face_embedding,
                    enrollment_image,
                    status
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s::jsonb,
                    %s,
                    'ACTIVE'
                )
                """,
                (
                    driver_id,
                    name,
                    address,
                    age,
                    json.dumps(embedding),
                    image_path
                )
            )

        conn.commit()


        # -------------------------------------------------
        # SUCCESS
        # -------------------------------------------------

        return {
            "status": "success",
            "message": "Driver enrolled successfully",
            "driver_id": driver_id,
            "driver_name": name,
            "embedding_dimension": ai_result[
                "embedding_dimension"
            ],
            "face_count": ai_result[
                "face_count"
            ],
            "enrollment_image": image_path
        }


    except HTTPException:

        conn.rollback()
        raise


    except Exception as e:

        conn.rollback()

        print(
            "Driver enrollment error:",
            e
        )

        raise HTTPException(
            status_code=500,
            detail="Driver enrollment failed"
        )


    finally:

        conn.close()

# =====================================================
# VEHICLE STATUS
# =====================================================

@router.post("/vehicles/{vehicle_id}/status")
def update_vehicle_status(
    vehicle_id: int,
    active: bool,
    esp_id: str = None
):
    """
    ESP32 updates vehicle ACTIVE / INACTIVE status.
    """

    conn = get_connection()

    try:

        with conn.cursor() as cur:

            if esp_id:
                cur.execute(
                    """
                    UPDATE vehicles
                    SET
                        active = %s,
                        esp_id = %s,
                        last_seen = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE vehicle_id = %s
                    RETURNING vehicle_id, active, esp_id, last_seen
                    """,
                    (
                        active,
                        esp_id,
                        vehicle_id
                    )
                )
            else:
                cur.execute(
                    """
                    UPDATE vehicles
                    SET
                        active = %s,
                        last_seen = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE vehicle_id = %s
                    RETURNING vehicle_id, active, esp_id, last_seen
                    """,
                    (
                        active,
                        vehicle_id
                    )
                )

            vehicle = cur.fetchone()

            if not vehicle:
                raise HTTPException(
                    status_code=404,
                    detail="Vehicle not found"
                )

        conn.commit()

        return {
            "status": "success",
            "vehicle_id": vehicle[0],
            "active": vehicle[1],
            "esp_id": vehicle[2],
            "last_seen": vehicle[3]
        }

    except HTTPException:
        conn.rollback()
        raise

    except Exception as e:
        conn.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        conn.close()


# =====================================================
# GET VEHICLES
# =====================================================

@router.get("/vehicles")
def get_vehicles():

    conn = get_connection()

    try:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    vehicle_id,
                    esp_id,
                    number_plate,
                    active,
                    status,
                    last_seen
                FROM vehicles
                ORDER BY vehicle_id
                """
            )

            rows = cur.fetchall()

            vehicles = []

            for row in rows:

                vehicles.append({
                    "vehicle_id": row[0],
                    "esp_id": row[1],
                    "number_plate": row[2],
                    "active": row[3],
                    "status": row[4],
                    "last_seen": row[5]
                })

        return vehicles

    finally:
        conn.close()


# =====================================================
# FACE VERIFICATION
# =====================================================

@router.post("/verification/verify")
def verify_driver(data: dict):

    live_embedding = data.get("face_embedding")
    vehicle_id = data.get("vehicle_id")
    ai_evidence = data.get("ai_evidence", {})

    if not live_embedding:
        raise HTTPException(
            status_code=400,
            detail="Face embedding is required"
        )

    if vehicle_id is None:
        raise HTTPException(
            status_code=400,
            detail="Vehicle ID is required"
        )

    try:
        live_embedding = np.asarray(
            live_embedding,
            dtype=np.float32
        )

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid face embedding"
        )

    if live_embedding.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty face embedding"
        )

    conn = get_connection()

    try:

        with conn.cursor() as cur:

            # -------------------------------------------------
            # Get registered drivers
            # -------------------------------------------------

            cur.execute(
                """
                SELECT
                    driver_id,
                    name,
                    face_embedding
                FROM drivers
                WHERE status = 'ACTIVE'
                  AND face_embedding IS NOT NULL
                """
            )

            drivers = cur.fetchall()

            best_driver = None
            best_similarity = -1.0

            # -------------------------------------------------
            # Compare live embedding with database embeddings
            # -------------------------------------------------

            for driver in drivers:

                driver_id = driver[0]
                name = driver[1]
                stored_embedding = driver[2]

                try:

                    stored_embedding = np.asarray(
                        stored_embedding,
                        dtype=np.float32
                    )

                    if stored_embedding.shape != live_embedding.shape:
                        continue

                    denominator = (
                        np.linalg.norm(live_embedding)
                        *
                        np.linalg.norm(stored_embedding)
                    )

                    if denominator == 0:
                        continue

                    similarity = float(
                        np.dot(
                            live_embedding,
                            stored_embedding
                        )
                        / denominator
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_driver = {
                            "driver_id": driver_id,
                            "name": name
                        }

                except Exception:
                    continue

            # -------------------------------------------------
            # Verification threshold
            # -------------------------------------------------

            MATCH_THRESHOLD = 0.40

            matched = (
                best_driver is not None
                and best_similarity >= MATCH_THRESHOLD
            )

            # -------------------------------------------------
            # Liveness
            # -------------------------------------------------

            liveness = ai_evidence.get(
                "liveness",
                "UNKNOWN"
            )

            liveness_passed = (
                str(liveness).upper()
                in ["LIVE", "REAL", "PASS"]
            )

            # -------------------------------------------------
            # Final authorization
            # -------------------------------------------------

            final_authorization = (
                matched
                and liveness_passed
            )

            if final_authorization:
                decision = "VERIFIED"
            else:
                decision = "REJECTED"

            driver_id = (
                best_driver["driver_id"]
                if matched
                else None
            )

            driver_name = (
                best_driver["name"]
                if matched
                else None
            )

            # -------------------------------------------------
            # Save verification result
            # -------------------------------------------------

            cur.execute(
                """
                INSERT INTO verification_results
                (
                    timestamp,
                    vehicle_id,
                    driver_id,
                    driver_name,
                    driver_confidence,
                    matched,
                    liveness,
                    liveness_confidence,
                    ai_evidence,
                    final_authorization,
                    decision
                )
                VALUES
                (
                    CURRENT_TIMESTAMP,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s::jsonb,
                    %s,
                    %s
                )
                RETURNING id
                """,
                (
                    vehicle_id,
                    driver_id,
                    driver_name,
                    best_similarity if matched else 0,
                    matched,
                    liveness,
                    ai_evidence.get(
                        "liveness_confidence",
                        0
                    ),
                    json.dumps(ai_evidence),
                    final_authorization,
                    decision
                )
            )

            record_id = cur.fetchone()[0]

        conn.commit()

        return {
            "status": "success",
            "decision": decision,
            "authorized": final_authorization,
            "driver_id": driver_id,
            "driver_name": driver_name,
            "similarity": round(
                best_similarity,
                4
            ) if best_driver else 0,
            "record_id": record_id
        }

    except HTTPException:
        conn.rollback()
        raise

    except Exception as e:
        conn.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        conn.close()