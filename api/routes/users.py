import asyncio
import base64
from datetime import datetime
from typing import Optional

import cv2
import face_recognition
import numpy as np
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.models.user import User
from app.services.attendance_service import AttendanceService

from typing import List
from fastapi import UploadFile, File, Form

class SingleUserImport(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    position: str = Field("Nhân viên", max_length=50)
    face_image_base64: str

class BatchImportBase64Request(BaseModel):
    users: List[SingleUserImport]

class BatchImportResponse(BaseModel):
    total: int
    success_count: int
    failed_count: int
    success_users: List[dict]
    errors: List[dict]

router = APIRouter()


class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    position: str = Field("Nhân viên", max_length=50)
    face_image_base64: str

class UserResponse(BaseModel):
    id: int
    name: str
    position: str
    created_at: str

def format_datetime(val) -> str:
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val) if val else datetime.now().isoformat()

# Trích xuất Encoding VÀ Crop ảnh khuôn mặt để validate
def _extract_and_validate_face(face_image_base64: str):
    if ',' in face_image_base64:
        image_data = face_image_base64.split(',')[1]
    else:
        image_data = face_image_base64

    try:
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Mã hóa Base64 không hợp lệ")

    if frame is None:
        raise HTTPException(status_code=400, detail="Định dạng ảnh không hợp lệ")

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    if not face_locations:
        raise HTTPException(status_code=400, detail="Không phát hiện khuôn mặt")
    if len(face_locations) > 1:
        raise HTTPException(status_code=400, detail="Phát hiện nhiều khuôn mặt, vui lòng chỉ chụp 1 người")

    # Crop vùng khuôn mặt để kiểm tra chất lượng
    top, right, bottom, left = face_locations[0]
    pad = 20
    crop_top = max(0, top - pad)
    crop_bottom = min(h, bottom + pad)
    crop_left = max(0, left - pad)
    crop_right = min(w, right + pad)
    crop_face = frame[crop_top:crop_bottom, crop_left:crop_right]

    # Kiểm tra chất lượng góc mặt / mờ nhòe / kích thước
    is_valid, msg = AttendanceService.validate_face_quality(crop_face)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Đăng ký thất bại: {msg}")

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    return face_encodings[0]


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate, request: Request):
    """Đăng ký nhân viên mới từ ảnh"""
    try:
        # 1. Trích xuất encoding & Validate chất lượng mặt trên ThreadPool riêng
        face_encoding = await asyncio.to_thread(_extract_and_validate_face, user_data.face_image_base64)

        engine = request.app.state.engine

        # 2. Kiểm tra trùng lặp khuôn mặt
        matched_id, score = engine.face_recognizer.recognize(None, face_encoding)
        if matched_id:
            existing_user = engine.face_recognizer.user_map.get(matched_id)
            user_name = existing_user.name if existing_user else f"ID {matched_id}"
            raise HTTPException(
                status_code=400,
                detail=f"Khuôn mặt đã được đăng ký bởi: {user_name}"
            )

        # 3. Lưu user vào DB
        def _save_to_db():
            db = request.app.state.db
            user_repo = request.app.state.user_repo

            new_user = User(
                id=None,
                name=user_data.name.strip(),
                position=user_data.position,
                face_encoding=User.encode_embedding(face_encoding)
            )
            with db.transaction() as conn:
                return user_repo.create(conn, new_user)

        user_id = await asyncio.to_thread(_save_to_db)

        # 4. Reload cache hệ thống nhận diện
        engine.reload_users()

        return UserResponse(
            id=user_id,
            name=user_data.name.strip(),
            position=user_data.position,
            created_at=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đăng ký: {str(e)}")


@router.get("/list")
def list_users(request: Request, limit: int = 100, offset: int = 0):
    """Lấy danh sách tất cả nhân viên (Hỗ trợ MySQL)"""
    db = request.app.state.db

    with db.transaction() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, name, position, created_at FROM users ORDER BY id LIMIT %s OFFSET %s",
                (limit, offset)
            )
            users = cursor.fetchall()

            cursor.execute("SELECT COUNT(*) as count FROM users")
            total = cursor.fetchone()["count"]

            return {
                "total": total,
                "limit": limit,
                "offset": offset,
                "users": [
                    {
                        "id": u["id"],
                        "name": u["name"],
                        "position": u["position"],
                        "created_at": format_datetime(u["created_at"])
                    }
                    for u in users
                ]
            }

@router.post("/import-files", response_model=BatchImportResponse)
async def import_users_from_files(
    files: List[UploadFile] = File(...), 
    request: Request = None
):
    """Upload trực tiếp 1 hoặc nhiều file ảnh để đăng ký nhanh (Đã sửa lỗi nhận nhầm trùng mặt)"""
    engine = request.app.state.engine
    db = request.app.state.db
    user_repo = request.app.state.user_repo

    success_users = []
    errors = []

    for file in files:
        # Tách tên và chức vụ từ tên file
        filename_without_ext = file.filename.rsplit('.', 1)[0]
        parts = filename_without_ext.split('_')
        name = parts[0].strip()
        position = parts[1].strip() if len(parts) > 1 else "Nhân viên"

        try:
            # Read file bytes & decode OpenCV
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                errors.append({"file": file.filename, "reason": "Định dạng ảnh không hợp lệ"})
                continue

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")

            if not face_locations:
                errors.append({"file": file.filename, "reason": "Không tìm thấy khuôn mặt"})
                continue
            if len(face_locations) > 1:
                errors.append({"file": file.filename, "reason": "Có nhiều hơn 1 khuôn mặt trong ảnh"})
                continue

            # Crop & Validate chất lượng mặt
            top, right, bottom, left = face_locations[0]
            pad = 20
            crop_face = frame[max(0, top-pad):min(h, bottom+pad), max(0, left-pad):min(w, right+pad)]
            is_valid, msg = AttendanceService.validate_face_quality(crop_face)
            if not is_valid:
                errors.append({"file": file.filename, "reason": f"Chất lượng ảnh kém: {msg}"})
                continue

            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]

            # Check trùng lặp với danh sách ĐÃ TỒN TẠI trong DB
            matched_id, score = engine.face_recognizer.recognize(None, face_encoding)
            if matched_id:
                existing_user = engine.face_recognizer.user_map.get(matched_id)
                u_name = existing_user.name if existing_user else f"ID {matched_id}"
                errors.append({"file": file.filename, "reason": f"Khuôn mặt đã trùng với: {u_name}"})
                continue

            # Save MySQL
            def _save():
                new_user = User(
                    id=None,
                    name=name,
                    position=position,
                    face_encoding=User.encode_embedding(face_encoding)
                )
                with db.transaction() as conn:
                    return user_repo.create(conn, new_user)

            user_id = await asyncio.to_thread(_save)

            # 🛑 ĐÃ XÓA DÒNG add_user TẠM THỜI GÂY LỖI CACHE TẠI ĐÂY 🛑

            success_users.append({"id": user_id, "name": name, "position": position, "file": file.filename})

        except Exception as e:
            errors.append({"file": file.filename, "reason": str(e)})

    # Reload toàn bộ cache từ MySQL chuẩn xác sau khi đã import xong tất cả
    engine.reload_users()

    return BatchImportResponse(
        total=len(files),
        success_count=len(success_users),
        failed_count=len(errors),
        success_users=success_users,
        errors=errors
    ) 

@router.delete("/{user_id}")
def delete_user(user_id: int, request: Request):
    """Xóa nhân viên"""
    db = request.app.state.db
    user_repo = request.app.state.user_repo
    user_name = None

    with db.transaction() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, name FROM users WHERE id = %s", (user_id,))
            user = cursor.fetchone()

            if user:
                user_name = user["name"]
                user_repo.delete(conn, user_id)

    if not user_name:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy nhân viên ID {user_id}")

    request.app.state.engine.reload_users()

    return {"message": f"Đã xóa nhân viên {user_name} (ID: {user_id}) thành công"}