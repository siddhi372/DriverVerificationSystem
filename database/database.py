import os
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv


# =====================================================
# LOAD ENVIRONMENT
# =====================================================

load_dotenv()


# =====================================================
# POSTGRESQL CONFIGURATION
# =====================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# =====================================================
# DATABASE CONNECTION
# =====================================================

def get_connection():

    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


# =====================================================
# CREATE TABLES
# =====================================================

def create_tables():

    connection = get_connection()
    cursor = connection.cursor()

    # -------------------------------------------------
    # VERIFICATION RESULTS
    # -------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS verification_results (

            id SERIAL PRIMARY KEY,

            timestamp TIMESTAMP NOT NULL,

            session_id VARCHAR(100),

            face_detected BOOLEAN,

            face_count INTEGER,

            image_quality VARCHAR(50),

            liveness VARCHAR(50),

            liveness_confidence DOUBLE PRECISION,

            driver_id VARCHAR(100),

            driver_name VARCHAR(100),

            driver_confidence DOUBLE PRECISION,

            matched BOOLEAN,

            estimated_age INTEGER,

            decision VARCHAR(50),

            vehicle_status VARCHAR(50),

            processing_time DOUBLE PRECISION

        )
    """)

    # -------------------------------------------------
    # ADMIN USERS
    # -------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admin_users (

            id SERIAL PRIMARY KEY,

            admin_id VARCHAR(100) UNIQUE NOT NULL,

            password_hash TEXT NOT NULL,

            name VARCHAR(100) NOT NULL,

            role VARCHAR(30) NOT NULL DEFAULT 'OWNER',

            is_active BOOLEAN NOT NULL DEFAULT TRUE,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            last_login TIMESTAMP

        )
    """)

    # -------------------------------------------------
    # DRIVERS
    # -------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drivers (

            id SERIAL PRIMARY KEY,

            driver_id VARCHAR(100) UNIQUE NOT NULL,

            name VARCHAR(100) NOT NULL,

            status VARCHAR(30) NOT NULL DEFAULT 'ACTIVE',

            assigned_vehicle VARCHAR(100),

            photo_path TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)

    # -------------------------------------------------
    # VEHICLES
    # -------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            vehicle_id SERIAL PRIMARY KEY,
            esp_id VARCHAR(100) UNIQUE NOT NULL,
            number_plate VARCHAR(50) UNIQUE NOT NULL,
            status VARCHAR(30) NOT NULL DEFAULT 'ACTIVE',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # -------------------------------------------------
    # DRIVER <-> VEHICLE MAPPING
    # -------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS driver_vehicle (
            id SERIAL PRIMARY KEY,
            driver_id VARCHAR(100) NOT NULL,
            vehicle_id INTEGER NOT NULL,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            CONSTRAINT fk_driver_vehicle_vehicle
                FOREIGN KEY (vehicle_id)
                REFERENCES vehicles(vehicle_id)
                ON DELETE CASCADE,

            CONSTRAINT uq_driver_vehicle
                UNIQUE (driver_id, vehicle_id)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_vehicles_esp_id
        ON vehicles(esp_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_vehicles_number_plate
        ON vehicles(number_plate)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_driver_vehicle_driver
        ON driver_vehicle(driver_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_driver_vehicle_vehicle
        ON driver_vehicle(vehicle_id)
    """)

    # -------------------------------------------------
    # MIGRATION FOR EXISTING DATABASE
    # -------------------------------------------------

    cursor.execute("""
        ALTER TABLE verification_results
        ADD COLUMN IF NOT EXISTS driver_id VARCHAR(100)
    """)

    cursor.execute("""
        ALTER TABLE verification_results
        ADD COLUMN IF NOT EXISTS estimated_age INTEGER
    """)

    cursor.execute("""
        ALTER TABLE verification_results
        ADD COLUMN IF NOT EXISTS processing_time DOUBLE PRECISION
    """)

    connection.commit()

    cursor.close()
    connection.close()

    print("PostgreSQL tables created successfully.")


# =====================================================
# SAVE VERIFICATION RESULT
# =====================================================

def save_verification_result(data):

    connection = get_connection()
    cursor = connection.cursor()

    # -------------------------------------------------
    # FACE DETECTION
    # -------------------------------------------------

    face_detection = data.get(
        "face_detection",
        {}
    )

    face_detected = face_detection.get(
        "detected",
        False
    )

    face_count = face_detection.get(
        "face_count",
        0
    )

    # -------------------------------------------------
    # IMAGE QUALITY
    # -------------------------------------------------

    image_quality_data = data.get(
        "image_quality",
        {}
    )

    image_quality = image_quality_data.get(
        "status",
        "UNKNOWN"
    )

    # -------------------------------------------------
    # LIVENESS
    # -------------------------------------------------

    liveness_data = data.get(
        "liveness",
        {}
    )

    liveness = liveness_data.get(
        "status",
        "UNKNOWN"
    )

    liveness_confidence = liveness_data.get(
        "confidence",
        0.0
    )

    # -------------------------------------------------
    # DRIVER AUTHENTICATION
    # -------------------------------------------------

    driver_data = data.get(
        "driver_authentication",
        {}
    )

    driver_id = driver_data.get(
        "driver_id"
    )

    driver_name = driver_data.get(
        "driver_name",
        "UNKNOWN"
    )

    driver_confidence = driver_data.get(
        "confidence",
        0.0
    )

    matched = driver_data.get(
        "matched",
        False
    )

    # -------------------------------------------------
    # AGE
    # -------------------------------------------------

    estimated_age = data.get(
        "estimated_age"
    )

    # Also support nested age data if used later
    if estimated_age is None:

        age_data = data.get(
            "age_estimation",
            {}
        )

        if isinstance(age_data, dict):

            estimated_age = age_data.get(
                "age"
            )

    # -------------------------------------------------
    # DECISION
    # -------------------------------------------------

    decision_data = data.get(
        "decision",
        {}
    )

    if isinstance(decision_data, dict):

        decision = decision_data.get(
            "result",
            "DENY"
        )

        vehicle_status = decision_data.get(
            "vehicle_status",
            "LOCKED"
        )

    else:

        decision = str(
            decision_data
        )

        vehicle_status = data.get(
            "vehicle_status",
            "LOCKED"
        )

    # -------------------------------------------------
    # PROCESSING TIME
    # -------------------------------------------------

    processing_time = data.get(
        "processing_time"
    )

    # -------------------------------------------------
    # TIMESTAMP
    # -------------------------------------------------

    timestamp = data.get(
        "timestamp"
    )

    if timestamp:

        try:

            timestamp = datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")
            )

        except (ValueError, TypeError):

            timestamp = datetime.now()

    else:

        timestamp = datetime.now()

    # -------------------------------------------------
    # SESSION ID
    # -------------------------------------------------

    session_id = data.get(
        "session_id"
    )

    # -------------------------------------------------
    # INSERT
    # -------------------------------------------------

    cursor.execute("""
        INSERT INTO verification_results (

            timestamp,
            session_id,
            face_detected,
            face_count,
            image_quality,
            liveness,
            liveness_confidence,
            driver_id,
            driver_name,
            driver_confidence,
            matched,
            estimated_age,
            decision,
            vehicle_status,
            processing_time

        )

        VALUES (

            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s

        )

        RETURNING id

    """, (

        timestamp,
        session_id,
        face_detected,
        face_count,
        image_quality,
        liveness,
        liveness_confidence,
        driver_id,
        driver_name,
        driver_confidence,
        matched,
        estimated_age,
        decision,
        vehicle_status,
        processing_time

    ))

    record_id = cursor.fetchone()[0]

    connection.commit()

    cursor.close()
    connection.close()

    return record_id


# =====================================================
# GET VERIFICATION HISTORY
# =====================================================

def get_verification_history(limit=50):

    connection = get_connection()

    cursor = connection.cursor(
        cursor_factory=RealDictCursor
    )

    cursor.execute("""
        SELECT

            id,
            timestamp,
            session_id,
            face_detected,
            face_count,
            image_quality,
            liveness,
            liveness_confidence,
            driver_id,
            driver_name,
            driver_confidence,
            matched,
            estimated_age,
            decision,
            vehicle_status,
            processing_time

        FROM verification_results

        ORDER BY id DESC

        LIMIT %s

    """, (limit,))

    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    return [
        dict(row)
        for row in rows
    ]


# =====================================================
# VERIFICATION SUMMARY
# =====================================================

def get_verification_summary():

    connection = get_connection()
    cursor = connection.cursor()

    # Total
    cursor.execute("""
        SELECT COUNT(*)
        FROM verification_results
    """)

    total_verifications = cursor.fetchone()[0]

    # Allowed
    cursor.execute("""
        SELECT COUNT(*)
        FROM verification_results
        WHERE decision IN ('ALLOW', 'VERIFIED')
    """)

    allowed = cursor.fetchone()[0]

    # Denied
    cursor.execute("""
        SELECT COUNT(*)
        FROM verification_results
        WHERE decision IN ('DENY', 'UNKNOWN', 'SPOOF')
    """)

    denied = cursor.fetchone()[0]

    # Live
    cursor.execute("""
        SELECT COUNT(*)
        FROM verification_results
        WHERE liveness = 'LIVE'
    """)

    live_attempts = cursor.fetchone()[0]

    # Spoof
    cursor.execute("""
        SELECT COUNT(*)
        FROM verification_results
        WHERE liveness = 'SPOOF'
    """)

    spoof_attempts = cursor.fetchone()[0]

    cursor.close()
    connection.close()

    return {

        "total_verifications":
            total_verifications,

        "allowed":
            allowed,

        "denied":
            denied,

        "live_attempts":
            live_attempts,

        "spoof_attempts":
            spoof_attempts

    }


# =====================================================
# CREATE ADMIN TABLE
# =====================================================

def create_admin_table():

    create_tables()

    print(
        "Admin users table ready."
    )


# =====================================================
# GET ADMIN BY ID
# =====================================================

def get_admin_by_id(admin_id):

    connection = get_connection()

    cursor = connection.cursor(
        cursor_factory=RealDictCursor
    )

    cursor.execute("""
        SELECT

            id,
            admin_id,
            password_hash,
            name,
            role,
            is_active,
            created_at,
            last_login

        FROM admin_users

        WHERE admin_id = %s

    """, (admin_id,))

    admin = cursor.fetchone()

    cursor.close()
    connection.close()

    if admin is None:
        return None

    return dict(admin)


# =====================================================
# CREATE ADMIN USER
# =====================================================

def create_admin_user(
    admin_id,
    password_hash,
    name,
    role="OWNER"
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO admin_users (

            admin_id,
            password_hash,
            name,
            role

        )

        VALUES (

            %s,
            %s,
            %s,
            %s

        )

        RETURNING id

    """, (
        admin_id,
        password_hash,
        name,
        role
    ))

    admin_id_db = cursor.fetchone()[0]

    connection.commit()

    cursor.close()
    connection.close()

    return admin_id_db


# =====================================================
# UPDATE LAST LOGIN
# =====================================================

def update_admin_last_login(admin_id):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        UPDATE admin_users

        SET last_login = CURRENT_TIMESTAMP

        WHERE admin_id = %s

    """, (admin_id,))

    connection.commit()

    cursor.close()
    connection.close()


# =====================================================
# DRIVER MANAGEMENT
# =====================================================

def add_driver(
    driver_id,
    name,
    assigned_vehicle=None
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO drivers (

            driver_id,
            name,
            status,
            assigned_vehicle

        )

        VALUES (

            %s,
            %s,
            'ACTIVE',
            %s

        )

        RETURNING id

    """, (
        driver_id,
        name,
        assigned_vehicle
    ))

    database_id = cursor.fetchone()[0]

    connection.commit()

    cursor.close()
    connection.close()

    return database_id


# =====================================================
# GET ALL DRIVERS
# =====================================================

def get_all_drivers():

    connection = get_connection()

    cursor = connection.cursor(
        cursor_factory=RealDictCursor
    )

    cursor.execute("""
        SELECT

            id,
            driver_id,
            name,
            status,
            assigned_vehicle,
            photo_path,
            created_at

        FROM drivers

        ORDER BY id DESC

    """)

    drivers = cursor.fetchall()

    cursor.close()
    connection.close()

    return [
        dict(driver)
        for driver in drivers
    ]


# =====================================================
# GET DRIVER BY ID
# =====================================================

def get_driver(driver_id):

    connection = get_connection()

    cursor = connection.cursor(
        cursor_factory=RealDictCursor
    )

    cursor.execute("""
        SELECT

            id,
            driver_id,
            name,
            status,
            assigned_vehicle,
            photo_path,
            created_at

        FROM drivers

        WHERE driver_id = %s

    """, (driver_id,))

    driver = cursor.fetchone()

    cursor.close()
    connection.close()

    if driver is None:
        return None

    return dict(driver)


# =====================================================
# UPDATE DRIVER STATUS
# =====================================================

def update_driver_status(
    driver_id,
    status
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        UPDATE drivers

        SET status = %s

        WHERE driver_id = %s

    """, (
        status,
        driver_id
    ))

    connection.commit()

    cursor.close()
    connection.close()


# =====================================================
# UPDATE DRIVER VEHICLE
# =====================================================

def update_driver_vehicle(
    driver_id,
    assigned_vehicle
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        UPDATE drivers

        SET assigned_vehicle = %s

        WHERE driver_id = %s

    """, (
        assigned_vehicle,
        driver_id
    ))

    connection.commit()

    cursor.close()
    connection.close()
def migrate_driver_verification_schema():
    """
    Adds the fields required for:
    - Driver face enrollment
    - Vehicle active/inactive tracking
    - Verification capture images
    - AI evidence
    - Final authorization
    """

    conn = get_connection()

    try:
        with conn.cursor() as cur:

            # =====================================================
            # DRIVERS
            # =====================================================

            cur.execute("""
                ALTER TABLE drivers
                ADD COLUMN IF NOT EXISTS address TEXT;
            """)

            cur.execute("""
                ALTER TABLE drivers
                ADD COLUMN IF NOT EXISTS age INTEGER;
            """)

            cur.execute("""
                ALTER TABLE drivers
                ADD COLUMN IF NOT EXISTS face_embedding JSONB;
            """)

            cur.execute("""
                ALTER TABLE drivers
                ADD COLUMN IF NOT EXISTS enrollment_image TEXT;
            """)

            # =====================================================
            # VEHICLES
            # =====================================================

            cur.execute("""
                ALTER TABLE vehicles
                ADD COLUMN IF NOT EXISTS active BOOLEAN DEFAULT FALSE;
            """)

            cur.execute("""
                ALTER TABLE vehicles
                ADD COLUMN IF NOT EXISTS last_seen TIMESTAMP;
            """)

            # =====================================================
            # VERIFICATION RESULTS
            # =====================================================

            cur.execute("""
                ALTER TABLE verification_results
                ADD COLUMN IF NOT EXISTS vehicle_id INTEGER;
            """)

            cur.execute("""
                ALTER TABLE verification_results
                ADD COLUMN IF NOT EXISTS captured_image TEXT;
            """)

            cur.execute("""
                ALTER TABLE verification_results
                ADD COLUMN IF NOT EXISTS ai_evidence JSONB;
            """)

            cur.execute("""
                ALTER TABLE verification_results
                ADD COLUMN IF NOT EXISTS final_authorization BOOLEAN DEFAULT FALSE;
            """)

            # =====================================================
            # FOREIGN KEY
            # =====================================================

            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_constraint
                        WHERE conname = 'fk_verification_vehicle'
                    ) THEN

                        ALTER TABLE verification_results
                        ADD CONSTRAINT fk_verification_vehicle
                        FOREIGN KEY (vehicle_id)
                        REFERENCES vehicles(vehicle_id)
                        ON DELETE SET NULL;

                    END IF;
                END
                $$;
            """)

        conn.commit()

        print("Database migration completed successfully.")

    except Exception as e:
        conn.rollback()
        print("Database migration failed:", e)
        raise

    finally:
        conn.close()

# =====================================================
# MAIN TEST
# =====================================================

if __name__ == "__main__":

    create_tables()

    print(
        "Database initialization complete."
    )

if __name__ == "__main__":
    migrate_driver_verification_schema()