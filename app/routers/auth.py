from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt

from database.database import get_db
from database.models import User


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


# ============================================================
# PASSWORD HASHING
# ============================================================

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


# ============================================================
# JWT SETTINGS
# ============================================================

SECRET_KEY = "SAFEHER-AI-DEMO-SECRET-CHANGE-LATER"
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60


# ============================================================
# REQUEST MODELS
# ============================================================

class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    phone: str | None = None
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# ============================================================
# PASSWORD FUNCTIONS
# ============================================================

def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(
        plain_password,
        hashed_password
    )


# ============================================================
# JWT TOKEN
# ============================================================

def create_access_token(user_id: int):

    expire = datetime.utcnow() + timedelta(
        minutes=TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": str(user_id),
        "exp": expire
    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


# ============================================================
# REGISTER
# ============================================================

@router.post("/register")
def register(
    user_data: RegisterRequest,
    db: Session = Depends(get_db)
):

    # Check whether email already exists
    existing_user = (
        db.query(User)
        .filter(User.email == user_data.email)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="An account with this email already exists."
        )

    # Password validation
    if len(user_data.password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 6 characters."
        )

    # Create user
    user = User(
        full_name=user_data.full_name,
        email=user_data.email,
        phone=user_data.phone,
        password_hash=hash_password(
            user_data.password
        )
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "success": True,
        "message": "Account created successfully.",
        "user_id": user.id
    }


# ============================================================
# LOGIN
# ============================================================

@router.post("/login")
def login(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):

    # Find user
    user = (
        db.query(User)
        .filter(User.email == login_data.email)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    # Verify password
    if not verify_password(
        login_data.password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    # Create JWT
    token = create_access_token(user.id)

    return {
        "success": True,
        "message": "Login successful.",
        "access_token": token,
        "token_type": "bearer",

        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "email": user.email,
            "phone": user.phone
        }
    }