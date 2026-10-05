
"""
STEP 1 - Vehicle database setup

Creates:
    vehicles
    driver_vehicle

Existing tables are NOT deleted or modified destructively.

Run from the project root:
    python add_vehicles.py
"""

import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


# ---------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL was not found in .env"
    )


# ---------------------------------------------------------
# DATABASE
# ---------------------------------------------------------

connection = psycopg2.connect(
    DATABASE_URL
)

connection.autocommit = False


try:

    cursor = connection.cursor()

    # -----------------------------------------------------
    # VEHICLES
    # -----------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicles (
            vehicle_id SERIAL PRIMARY KEY,
            esp_id VARCHAR(100) UNIQUE NOT NULL,
            number_plate VARCHAR(50) UNIQUE NOT NULL,
            status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    # -----------------------------------------------------
    # DRIVER <-> VEHICLE
    # -----------------------------------------------------

    cursor.execute(
        """
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
        );
        """
    )

    # -----------------------------------------------------
    # INDEXES
    # -----------------------------------------------------

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_vehicles_esp_id
        ON vehicles(esp_id);
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_vehicles_number_plate
        ON vehicles(number_plate);
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_driver_vehicle_driver
        ON driver_vehicle(driver_id);
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_driver_vehicle_vehicle
        ON driver_vehicle(vehicle_id);
        """
    )

    connection.commit()

    print()
    print("=" * 60)
    print("VEHICLE DATABASE SETUP COMPLETE")
    print("=" * 60)
    print()
    print("Created / verified:")
    print("  ✓ vehicles")
    print("  ✓ driver_vehicle")
    print("  ✓ required indexes")
    print()

    # -----------------------------------------------------
    # SHOW CURRENT VEHICLES
    # -----------------------------------------------------

    cursor.execute(
        """
        SELECT
            vehicle_id,
            esp_id,
            number_plate,
            status
        FROM vehicles
        ORDER BY vehicle_id;
        """
    )

    vehicles = cursor.fetchall()

    if vehicles:

        print("CURRENT VEHICLES")
        print("-" * 60)

        for vehicle in vehicles:
            print(
                f"Vehicle ID: {vehicle[0]} | "
                f"ESP ID: {vehicle[1]} | "
                f"Plate: {vehicle[2]} | "
                f"Status: {vehicle[3]}"
            )

    else:

        print(
            "No vehicles registered yet."
        )

    print()
    print("=" * 60)

finally:

    cursor.close()
    connection.close()
