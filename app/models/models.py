from sqlalchemy import Column, Integer, String, Date, DateTime, Enum, LargeBinary, UniqueConstraint, Float, Index, ForeignKey
from sqlalchemy.sql import func
from ..database import Base
import enum

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    position = Column(String(100), nullable=False)
    face_encoding = Column(LargeBinary, nullable=False)   # binary numpy array
    face_model = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AttendanceStatus(enum.Enum):
    checked_in = "checked_in"
    checked_out = "checked_out"
    missing_checkout = "missing_checkout"


class AttendanceLog(Base):
    __tablename__ = "attendance_logs"
    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_user_date"),
        Index("idx_date", "date"),
        Index("idx_user_date", "user_id", "date"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    date = Column(Date, nullable=False)
    check_in = Column(DateTime(timezone=True), nullable=True)
    check_out = Column(DateTime(timezone=True), nullable=True)
    status = Column(Enum(AttendanceStatus), nullable=False)
    check_in_liveness_score = Column(Float, nullable=True)
    check_out_liveness_score = Column(Float, nullable=True)
    check_in_face_score = Column(Float, nullable=True)
    check_out_face_score = Column(Float, nullable=True)
