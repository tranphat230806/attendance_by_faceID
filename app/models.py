from sqlalchemy import Column, Integer, String, Date, DateTime, Enum, LargeBinary, UniqueConstraint
from sqlalchemy.sql import func
from .db import Base
import enum

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    position = Column(String(100), nullable=False)
    face_encoding = Column(LargeBinary, nullable=False)   # pickled numpy array
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AttendanceStatus(enum.Enum):
    checked_in = "checked_in"
    checked_out = "checked_out"


class AttendanceLog(Base):
    __tablename__ = "attendance_logs"
    __table_args__ = (UniqueConstraint("user_id", "date", name="uq_user_date"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    date = Column(Date, nullable=False)
    check_in = Column(DateTime(timezone=True), nullable=True)
    check_out = Column(DateTime(timezone=True), nullable=True)
    status = Column(Enum(AttendanceStatus), nullable=False)
