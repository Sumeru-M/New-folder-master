from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pymongo.errors import DuplicateKeyError

from auth_service.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    get_password_hash,
)
from auth_service.database import get_users_collection
from auth_service.models import Token, UserCreate, UserPublic


router = APIRouter(tags=["auth"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(user: UserCreate) -> dict:
    users = get_users_collection()

    existing = users.find_one(
        {"$or": [{"username": user.username}, {"email": str(user.email)}]},
        {"_id": 0, "username": 1, "email": 1},
    )
    if existing:
        if existing.get("username") == user.username:
            raise HTTPException(status_code=409, detail="Username is already registered")
        raise HTTPException(status_code=409, detail="Email is already registered")

    doc = {
        "username": user.username,
        "email": str(user.email),
        "hashed_password": get_password_hash(user.password),
        "created_at": datetime.now(timezone.utc),
    }

    try:
        users.insert_one(doc)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=409, detail="Username or email already registered") from exc

    return {"message": "User registered successfully"}


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user["username"]})
    return Token(access_token=access_token)


@router.get("/me", response_model=UserPublic)
def me(current_user: UserPublic = Depends(get_current_user)) -> UserPublic:
    return current_user
