from database.database import (
    create_tables,
    save_verification_result,
    get_verification_history,
    get_verification_summary
)


# =====================================================
# CREATE TABLE
# =====================================================

create_tables()

print("Database connected successfully.")


# =====================================================
# TEST VERIFICATION JSON
# =====================================================

test_data = {

    "session_id": "TEST_POSTGRES_001",

    "timestamp": "2026-08-15T20:30:00",

    "face_detection": {
        "detected": True,
        "face_count": 1
    },

    "image_quality": {
        "status": "GOOD"
    },

    "liveness": {
        "status": "LIVE",
        "confidence": 0.99
    },

    "driver_authentication": {
        "driver_name": "Siddhi",
        "confidence": 0.95,
        "matched": True
    },

    "decision": {
        "result": "ALLOW",
        "vehicle_status": "UNLOCKED"
    }
}


# =====================================================
# SAVE
# =====================================================

record_id = save_verification_result(
    test_data
)

print(
    "Record ID:",
    record_id
)


# =====================================================
# HISTORY
# =====================================================

history = get_verification_history()

print("\nVERIFICATION HISTORY:")

for record in history:

    print(record)


# =====================================================
# SUMMARY
# =====================================================

summary = get_verification_summary()

print("\nVERIFICATION SUMMARY:")

print(summary)