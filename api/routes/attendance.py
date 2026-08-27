import asyncio
import base64
from datetime import datetime, timedelta, time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

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

# Hàm format thời gian tương thích an toàn với MySQL (hỗ trợ cả timedelta)
def format_mysql_time(val) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, timedelta):
        # MySQL thường trả về timedelta cho cột kiểu TIME
        total_seconds = int(val.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    if isinstance(val, (datetime, time)):
        return val.strftime("%H:%M:%S")
    return str(val)

def _decode_and_process(image_base64: str, engine):
    if ',' in image_base64:
        image_data = image_base64.split(',')[1]
    else:
        image_data = image_base64
        
    try:
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Mã hóa Base64 không hợp lệ")

    if frame is None:
        raise HTTPException(status_code=400, detail="Định dạng ảnh không hợp lệ")

    return engine.process_frame_sync(frame)


@router.post("/process", response_model=AttendanceResponse)
async def process_attendance(data: FaceData, request: Request):
    """Xử lý điểm danh từ ảnh gửi lên"""
    try:
        engine = request.app.state.engine
        
        # Chạy tác vụ xử lý AI trên Thread riêng tránh block Event Loop
        result = await asyncio.to_thread(_decode_and_process, data.image_base64, engine)

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message"))

        return AttendanceResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")


# Đổi thành "def" đồng bộ để tránh block async event loop khi query DB
@router.get("/check-status/{user_id}")
def check_user_status(user_id: int, request: Request):
    """Kiểm tra trạng thái điểm danh của nhân viên"""
    db = request.app.state.db

    with db.transaction() as conn:
        with conn.cursor() as cursor:
            # Sửa dấu ? thành %s cho MySQL
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
    """Lấy lịch sử điểm danh hôm nay"""
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