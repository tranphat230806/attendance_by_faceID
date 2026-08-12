import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:phat2026%40@localhost/attendance",
)
FALLBACK_DATABASE_URL = "sqlite:///./attendance.db"


def init_engine():
    global engine, SessionLocal
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        with engine.connect():
            pass
    except Exception as exc:
        print(
            "WARNING: Could not connect to primary database, "
            f"falling back to SQLite ({FALLBACK_DATABASE_URL}): {exc}"
        )
        engine = create_engine(
            FALLBACK_DATABASE_URL,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False},
        )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()
init_engine()
