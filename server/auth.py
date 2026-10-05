from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import bcrypt
from jose import jwt

from database.database import (
    get_admin_by_id,
    update_admin_last_login,
    create_admin_user
)


# =====================================================
# CONFIGURATION
# =====================================================

SECRET_KEY = "CHANGE_THIS_TO_A_RANDOM_SECRET_KEY"

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60


# =====================================================
# ROUTER
# =====================================================

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


# =====================================================
# LOGIN REQUEST
# =====================================================

class LoginRequest(BaseModel):
    admin_id: str
    password: str


# =====================================================
# REGISTER REQUEST
# =====================================================

class RegisterRequest(BaseModel):
    admin_id: str
    name: str
    password: str


# =====================================================
# PASSWORD HASH
# =====================================================

def hash_password(password: str) -> str:

    password_bytes = password.encode("utf-8")

    hashed = bcrypt.hashpw(
        password_bytes,
        bcrypt.gensalt()
    )

    return hashed.decode("utf-8")


# =====================================================
# VERIFY PASSWORD
# =====================================================

def verify_password(
    plain_password: str,
    hashed_password: str
) -> bool:

    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8")
    )


# =====================================================
# CREATE JWT TOKEN
# =====================================================

def create_access_token(
    admin_id: str,
    role: str
):

    expire = (
        datetime.now(timezone.utc)
        + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
    )

    payload = {
        "sub": admin_id,
        "role": role,
        "exp": expire
    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


# =====================================================
# LOGIN API
# =====================================================

@router.post("/login")
def admin_login(login: LoginRequest):

    # -------------------------------------------------
    # FIND ADMIN
    # -------------------------------------------------

    admin = get_admin_by_id(
        login.admin_id
    )

    if admin is None:

        raise HTTPException(
            status_code=401,
            detail="Invalid admin ID or password"
        )


    # -------------------------------------------------
    # CHECK ACTIVE STATUS
    # -------------------------------------------------

    if not admin["is_active"]:

        raise HTTPException(
            status_code=403,
            detail="Admin account is disabled"
        )


    # -------------------------------------------------
    # VERIFY PASSWORD
    # -------------------------------------------------

    if not verify_password(
        login.password,
        admin["password_hash"]
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid admin ID or password"
        )


    # -------------------------------------------------
    # UPDATE LAST LOGIN
    # -------------------------------------------------

    update_admin_last_login(
        login.admin_id
    )


    # -------------------------------------------------
    # CREATE TOKEN
    # -------------------------------------------------

    token = create_access_token(
        admin_id=admin["admin_id"],
        role=admin["role"]
    )


    # -------------------------------------------------
    # RESPONSE
    # -------------------------------------------------

    return {

        "status": "success",

        "message": "Login successful",

        "access_token": token,

        "token_type": "bearer",

        "admin": {

            "admin_id":
                admin["admin_id"],

            "name":
                admin["name"],

            "role":
                admin["role"]
        }
    }


# =====================================================
# REGISTER API
# =====================================================

@router.post("/register")
def admin_register(
    register: RegisterRequest
):

    # -------------------------------------------------
    # CLEAN INPUT
    # -------------------------------------------------

    admin_id = register.admin_id.strip()

    name = register.name.strip()

    password = register.password


    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------

    if not admin_id:

        raise HTTPException(
            status_code=400,
            detail="Admin ID is required"
        )


    if not name:

        raise HTTPException(
            status_code=400,
            detail="Name is required"
        )


    if len(password) < 6:

        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 6 characters"
        )


    # -------------------------------------------------
    # CHECK EXISTING ADMIN
    # -------------------------------------------------

    existing_admin = get_admin_by_id(
        admin_id
    )

    if existing_admin is not None:

        raise HTTPException(
            status_code=409,
            detail="Admin ID already exists"
        )


    # -------------------------------------------------
    # HASH PASSWORD
    # -------------------------------------------------

    password_hash = hash_password(
        password
    )


    # -------------------------------------------------
    # CREATE ADMIN
    # -------------------------------------------------

    try:

        database_id = create_admin_user(

            admin_id=admin_id,

            password_hash=password_hash,

            name=name,

            role="ADMIN"
        )

    except Exception as e:

        print(
            "Admin registration error:",
            e
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to create admin account"
        )


    # -------------------------------------------------
    # RESPONSE
    # -------------------------------------------------

    return {

        "status": "success",

        "message":
            "Admin account created successfully",

        "admin": {

            "id":
                database_id,

            "admin_id":
                admin_id,

            "name":
                name,

            "role":
                "ADMIN"
        }
    }