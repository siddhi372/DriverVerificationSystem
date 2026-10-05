from database.database import (
    create_admin_table,
    create_admin_user
)

from server.auth import hash_password


# =====================================================
# OWNER DETAILS
# =====================================================

ADMIN_ID = "OWNER001"

PASSWORD = "owner123"

NAME = "System Owner"


# =====================================================
# CREATE TABLE
# =====================================================

create_admin_table()


# =====================================================
# HASH PASSWORD
# =====================================================

password_hash = hash_password(
    PASSWORD
)


# =====================================================
# CREATE OWNER
# =====================================================

try:

    admin_id = create_admin_user(

        admin_id=ADMIN_ID,

        password_hash=password_hash,

        name=NAME,

        role="OWNER"

    )

    print(
        "Owner account created successfully."
    )

    print(
        "Database ID:",
        admin_id
    )

    print(
        "Admin ID:",
        ADMIN_ID
    )

    print(
        "Password:",
        PASSWORD
    )


except Exception as e:

    print(
        "Could not create owner:"
    )

    print(e)