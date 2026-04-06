import os
from functools import lru_cache

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "auth_db")


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    return MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)


def get_database() -> Database:
    return get_client()[MONGODB_DB_NAME]


def get_users_collection() -> Collection:
    return get_database()["users"]


def init_db() -> None:
    users = get_users_collection()
    users.create_index([("username", ASCENDING)], unique=True)
    users.create_index([("email", ASCENDING)], unique=True)


def close_db() -> None:
    try:
        get_client().close()
    finally:
        get_client.cache_clear()
