import asyncio
import base64
import face_recognition
from datetime import datetime, timedelta, time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.services.attendance_service import AttendanceService

router = APIRouter()

class FaceData(BaseModel):
    image_base64: str = Field(..., description="Ảnh khuôn mặt mã hóa base64")
    user_id: Optional[int] = Field(None, description="ID nhân viên (tùy chọn)")

class AttendanceResponse(BaseModel):
    status: str
    message: str
    user_id: Optional[int] = None
    user_name: Optional[str] = None
    position: Optional[str] = None
    score: Optional[float] = None
    check_in: Optional[str] = None
    check_out: Optional[str] = None

def format_mysql_time(val) -> Optional[str]:
    """Format thời gian linh hoạt tương thích MySQL (datetime, time, timedelta, string)"""
    if val is None:
        return None
    if isinstance(val, timedelta):
        total_seconds = int(val.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    if isinstance(val, (datetime, time)):
        return val.strftime("%H:%M:%S")
    if isinstance(val, str):
        # Trích xuất phần thời gian HH:MM:SS nếu chuỗi là ISO format
        return val.split("T")[-1].split(".")[0] if "T" in val else val
    return str(val)

def decode_base64_image(base64_str: str) -> np.ndarray:
    """Giải mã chuỗi Base64 sang ảnh OpenCV (Mat) an toàn"""
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]

        base64_str = base64_str.replace(' ', '+')

        missing_padding = len(base64_str) % 4
        if missing_padding:
            base64_str += '=' * (4 - missing_padding)

        image_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Định dạng ảnh không hợp lệ")

        return frame

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Mã hóa Base64 không hợp lệ: {str(e)}")

# Hàm xử lý trích xuất vector khuôn mặt và gọi Service điểm danh
def _decode_and_process(base64_str: str, engine) -> dict:
    frame = decode_base64_image(base64_str)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. Phát hiện vị trí khuôn mặt
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    if not face_locations:
        return {"status": "error", "message": "Không phát hiện thấy khuôn mặt trong ảnh"}
    
    if len(face_locations) > 1:
        return {"status": "error", "message": "Phát hiện nhiều hơn 1 khuôn mặt, vui lòng chỉ đứng 1 người"}

    # 2. Crop vùng ảnh khuôn mặt để validate Liveness / Quality
    top, right, bottom, left = face_locations[0]
    pad = 20
    crop_face = frame[max(0, top - pad):min(h, bottom + pad), max(0, left - pad):min(w, right + pad)]

    # 3. Trích xuất vector embedding (face_encoding)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    if not face_encodings:
        return {"status": "error", "message": "Không thể trích xuất đặc trưng khuôn mặt"}

    face_encoding = face_encodings[0]

    # 4. Thực thi điểm danh từ AttendanceService
    result = engine.attendance_service.process_attendance(crop_face, face_encoding)

    # Lấy thông tin user nhận diện được từ memory map
    matched_id, score = engine.face_recognizer.recognize(None, face_encoding)
    user = engine.face_recognizer.user_map.get(matched_id) if matched_id else None

    return {
        "status": result.status,
        "message": result.message,
        "user_id": user.id if user else None,
        "user_name": user.name if user else None,
        "position": user.position if user else None,
        "score": round(score, 4) if score else None,
        "check_in": format_mysql_time(result.check_in),
        "check_out": format_mysql_time(result.check_out)
    }

@router.post("/process", response_model=AttendanceResponse)
async def process_attendance(data: FaceData, request: Request):
    """Xử lý điểm danh từ ảnh gửi lên"""
    try:
        engine = request.app.state.engine
        
        # Chạy tác vụ xử lý AI trên ThreadPool riêng để tránh block Event Loop
        result = await asyncio.to_thread(_decode_and_process, data.image_base64, engine)

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message"))

        return AttendanceResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

@router.get("/check-status/{user_id}")
def check_user_status(user_id: int, request: Request):
    """Kiểm tra trạng thái điểm danh hôm nay của nhân viên"""
    db = request.app.state.db

    with db.transaction() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT check_in, check_out, status 
                FROM attendance_logs 
                WHERE user_id = %s AND date = CURDATE()
                ORDER BY id DESC LIMIT 1
            """, (user_id,))
            
            result = cursor.fetchone()

            if not result:
                return {
                    "user_id": user_id,
                    "status": "not_checked_in",
                    "message": "Chưa điểm danh hôm nay"
                }

            return {
                "user_id": user_id,
                "status": result["status"],
                "check_in": format_mysql_time(result["check_in"]),
                "check_out": format_mysql_time(result["check_out"])
            }

@router.get("/today")
def get_today_attendance(request: Request):
    """Lấy toàn bộ lịch sử điểm danh hôm nay"""
    db = request.app.state.db

    with db.transaction() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT a.id, u.name, u.position, a.check_in, a.check_out, a.status
                FROM attendance_logs a
                JOIN users u ON a.user_id = u.id
                WHERE a.date = CURDATE()
                ORDER BY a.id DESC
            """)
            
            logs = cursor.fetchall()

            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "total": len(logs),
                "logs": [
                    {
                        "id": log["id"],
                        "user_name": log["name"],
                        "position": log["position"],
                        "check_in": format_mysql_time(log["check_in"]),
                        "check_out": format_mysql_time(log["check_out"]),
                        "status": log["status"]
                    }
                    for log in logs
                ]
            }