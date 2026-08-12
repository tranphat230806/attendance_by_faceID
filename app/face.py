import cv2
import base64
import numpy as np
import pickle
import math
import time
from typing import Tuple, Optional

import mediapipe as mp
import face_recognition

# ---------- Liveness & Face Mesh setup ----------
# Sửa cách khởi tạo MediaPipe Face Mesh tránh lỗi Syntax/Attribute
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]   # landmark indices
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


def _eye_aspect_ratio(landmarks, eye_idx, image_w, image_h) -> float:
    pts = np.array([(landmarks[i].x * image_w, landmarks[i].y * image_h) for i in eye_idx])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    ear = (A + B) / (2.0 * C + 1e-6)
    return ear


def is_live_blink(frame: np.ndarray, blink_threshold: float = 0.28) -> bool:
    """
    Kiểm tra liveness cho ảnh chụp từ nút bấm:
    1. Kiểm tra độ sắc nét/kết cấu da (Laplacian Variance) chống ảnh mờ/lóa màn hình.
    2. Kiểm tra sự tồn tại của khuôn mặt 3D và trạng thái mở mắt tự nhiên.
    """
    h, w = frame.shape[:2]
    
    # Kiểm tra Texture/Độ nét
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < 50.0:
        return False

    # Xử lý Face Mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return False
            
        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = _eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)
        right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)
        ear = (left_ear + right_ear) / 2.0
        
        return ear > 0.15


def get_face_encoding(frame: np.ndarray) -> Optional[np.ndarray]:
    """Trích xuất 128-d face embedding từ frame ảnh."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model='hog')
    if not locations:
        return None
    encodings = face_recognition.face_encodings(rgb, known_face_locations=locations)
    return encodings[0] if encodings else None


def match_face(encoding: np.ndarray, stored_encodings: list[bytes],
               tolerance: float = 0.45) -> Optional[int]:
    """So sánh vector khuôn mặt với danh sách lưu trong CSDL."""
    if encoding is None or not stored_encodings:
        return None

    for idx, blob in enumerate(stored_encodings):
        if not blob:
            continue

        db_enc = None
        try:
            db_enc = np.frombuffer(blob, dtype=np.float32)
        except Exception:
            pass

        if db_enc is None or db_enc.size == 0:
            try:
                decoded = pickle.loads(blob)
                if isinstance(decoded, np.ndarray):
                    db_enc = decoded
            except Exception:
                db_enc = None

        if db_enc is None or db_enc.size == 0:
            continue

        dist = np.linalg.norm(encoding - db_enc)
        if dist < tolerance:
            return idx
    return None


def decode_base64_image(b64_str: str) -> np.ndarray:
    """Giải mã ảnh base64 từ frontend canvas."""
    if not b64_str or len(b64_str) < 50:
        raise ValueError("Dữ liệu ảnh trống hoặc không hợp lệ.")
    header, data = b64_str.split(',', 1) if ',' in b64_str else (None, b64_str)
    try:
        img_bytes = base64.b64decode(data)
    except Exception:
        raise ValueError("Không thể giải mã base64. Vui lòng thử lại.")
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("Ảnh không hợp lệ hoặc bị hỏng. Vui lòng chụp lại.")
    return frame