from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from datetime import datetime, date
import json
import numpy as np

from .db import SessionLocal, engine, Base
from . import models, schemas, face

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
def startup_event():
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as exc:
        print(f"WARNING: could not create database tables on startup: {exc}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/attendance", response_model=schemas.AttendanceResult)
async def attendance(payload: schemas.FaceFrame, db: Session = Depends(get_db)):
    # 1️⃣ Decode incoming image
    try:
        frame = face.decode_base64_image(payload.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # 2️⃣ Liveness detection (blink). If you want other actions, extend here.
    if not face.is_live_blink(frame):
        return schemas.AttendanceResult(
            success=False,
            message="Liveness check failed – please blink."
        )

    # 3️⃣ Extract face embedding
    encoding = face.get_face_encoding(frame)
    if encoding is None:
        return schemas.AttendanceResult(
            success=False,
            message="No face detected."
        )

    # 4️⃣ Load all stored encodings (small company => load all; otherwise use indexing)
    users = db.query(models.User).all()
    stored_encodings = [u.face_encoding for u in users]
    match_idx = face.match_face(encoding, stored_encodings)

    if match_idx is None:
        return schemas.AttendanceResult(
            success=False,
            message="Face not recognized."
        )

    user = users[match_idx]

    # 5️⃣ Attendance logic
    today = date.today()
    log = (
        db.query(models.AttendanceLog)
        .filter(models.AttendanceLog.user_id == user.id, models.AttendanceLog.date == today)
        .first()
    )
    now = datetime.utcnow()

    if not log:
        # first time today → check‑in
        log = models.AttendanceLog(
            user_id=user.id,
            date=today,
            check_in=now,
            status=models.AttendanceStatus.checked_in,
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        message = "Check‑in successful."
    else:
        # already checked‑in → update check‑out (if not already)
        if log.check_out is None:
            log.check_out = now
            log.status = models.AttendanceStatus.checked_out
            db.commit()
            message = "Check‑out recorded."
        else:
            message = "Attendance already completed for today."

    # 6️⃣ Build response
    return schemas.AttendanceResult(
        success=True,
        message=message,
        name=user.name,
        position=user.position,
        check_in=log.check_in.isoformat() if log.check_in else None,
        check_out=log.check_out.isoformat() if log.check_out else None,
    )


@app.post("/api/register", response_model=schemas.RegistrationResult)
async def register(payload: schemas.RegisterFrame, db: Session = Depends(get_db)):
    if not payload.name.strip() or not payload.position.strip():
        return schemas.RegistrationResult(
            success=False,
            message="Tên và chức vụ không được để trống."
        )

    try:
        frame = face.decode_base64_image(payload.image)
    except ValueError as exc:
        return schemas.RegistrationResult(
            success=False,
            message=str(exc)
        )
    except Exception:
        return schemas.RegistrationResult(
            success=False,
            message="Dữ liệu hình ảnh không hợp lệ. Vui lòng thử lại."
        )

    encoding = face.get_face_encoding(frame)
    if encoding is None:
        return schemas.RegistrationResult(
            success=False,
            message="Không phát hiện khuôn mặt. Vui lòng thử lại."
        )

    users = db.query(models.User).all()
    stored_encodings = [u.face_encoding for u in users]
    if face.match_face(encoding, stored_encodings) is not None:
        return schemas.RegistrationResult(
            success=False,
            message="Khuôn mặt đã được đăng ký."
        )

    try:
        face_bytes = encoding.astype(np.float32).tobytes()
    except Exception:
        return schemas.RegistrationResult(
            success=False,
            message="Không thể trích xuất encoding khuôn mặt." 
        )

    user = models.User(
        name=payload.name.strip(),
        position=payload.position.strip(),
        face_encoding=face_bytes,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return schemas.RegistrationResult(
        success=True,
        message="Đăng ký thành công.",
        name=user.name,
        position=user.position,
    )
