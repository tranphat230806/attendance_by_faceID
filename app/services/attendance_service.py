import cv2
import numpy as np
from app.config import Config
from datetime import datetime, timedelta
from typing import Optional, NamedTuple, Tuple

from app.database import Database
from app.repositories.user_repository import UserRepository
from app.repositories.attendance_repository import AttendanceRepository
from app.services.liveness_service import RealLivenessDetector
from app.services.face_service import VectorFaceRecognizer

class AttendanceResult(NamedTuple):
    status: str
    message: str
    check_in: Optional[datetime] = None
    check_out: Optional[datetime] = None

# Helper hỗ trợ parse thời gian an toàn cho cả MySQL và SQLite
def parse_db_datetime(val) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val)
        except ValueError:
            return datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
    return None

class AttendanceService:
    """Service xử lý điểm danh và kiểm tra chất lượng khuôn mặt"""
    
    def __init__(
        self,
        db: Database,
        user_repo: UserRepository,
        attendance_repo: AttendanceRepository,
        liveness_detector: RealLivenessDetector,
        face_recognizer: VectorFaceRecognizer,
        cooldown_seconds: int = 300
    ):
        self.db = db
        self.user_repo = user_repo
        self.attendance_repo = attendance_repo
        self.liveness_detector = liveness_detector
        self.face_recognizer = face_recognizer
        self.cooldown_seconds = cooldown_seconds

    @staticmethod
    def validate_face_quality(face_image: np.ndarray) -> Tuple[bool, str]:
        """
        Ràng buộc chất lượng khuôn mặt chuẩn bảo mật cao
        """
        if face_image is None or face_image.size == 0:
            return False, "Không nhận diện được vùng ảnh khuôn mặt"

        h, w, _ = face_image.shape

        # 1. Ràng buộc kích thước khuôn mặt (Siết chặt khoảng cách đứng trước camera)
        min_size = getattr(Config, 'MIN_FACE_SIZE', 140)
        if h < min_size or w < min_size:
            return False, f"Khuôn mặt quá nhỏ ({w}x{h}px). Vui lòng ghé sát camera hơn (Tối thiểu {min_size}px)"

        # 2. Ràng buộc tỉ lệ khung hình (Chống nghiêng mặt, che khuất góc mặt)
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.70 or aspect_ratio > 1.30:
            return False, "Khuôn mặt bị nghiêng hoặc bị che một phần, vui lòng nhìn thẳng"

        # 3. Ràng buộc độ nét (Chống mờ nhòe do di chuyển)
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        min_blur = getattr(Config, 'MIN_LAPLACIAN_SCORE', 70.0)
        
        if blur_score < min_blur:
            return False, f"Ảnh bị mờ nhòe (Độ nét: {blur_score:.1f}/{min_blur}), vui lòng giữ yên đầu"

        return True, "Hợp lệ"
    def process_attendance(
        self, 
        face_image: np.ndarray, 
        face_encoding: np.ndarray,
        now: Optional[datetime] = None
    ) -> AttendanceResult:
        """Xử lý điểm danh có kiểm tra Liveness và Quality"""
        if now is None:
            now = datetime.now()

        # Step 1: Kiểm tra Ràng buộc chất lượng mặt
        is_valid_quality, quality_msg = self.validate_face_quality(face_image)
        if not is_valid_quality:
            return AttendanceResult(status="error", message=quality_msg)

        # Step 2: Kiểm tra Liveness (Chống ảnh tĩnh / Chống giả mạo)
        avg_ear, texture_score = self.liveness_detector.get_eye_aspect_ratio(face_image)
        if avg_ear is None:
            return AttendanceResult(
                status="error",
                message="Phát hiện ảnh chụp/điện thoại giả mạo hoặc phản xạ màn hình"
            )

        # Step 3: Nhận diện User
        with self.db.transaction() as conn:
            user_id, score = self.face_recognizer.recognize(conn, face_encoding)
            
            if not user_id:
                return AttendanceResult(
                    status="error",
                    message="Khuôn mặt chưa đăng ký trong hệ thống"
                )
            
            user = self.face_recognizer.user_map.get(user_id)
            if not user:
                return AttendanceResult(
                    status="error",
                    message="Dữ liệu người dùng không tồn tại"
                )
            
            # Step 4: Kiểm tra log hôm nay
            today_log = self.attendance_repo.get_today_log(conn, user_id)
            
            if today_log is None:
                # Check-in mới
                log_id = self.attendance_repo.create_log(conn, user_id, now)
                return AttendanceResult(
                    status="checked_in",
                    message=f"CHECK-IN THÀNH CÔNG: {user.name.upper()}",
                    check_in=now
                )
            
            check_in_dt = parse_db_datetime(today_log['check_in'])
            check_out_dt = parse_db_datetime(today_log['check_out'])

            # Đã check-out trước đó
            if today_log['status'] == 'checked_out':
                return AttendanceResult(
                    status="already_checked_out",
                    message=f"ĐÃ CHECK-OUT HÔM NAY: {user.name.upper()}",
                    check_in=check_in_dt,
                    check_out=check_out_dt
                )
            
            # Kiểm tra Cooldown an toàn trên MySQL
            if check_in_dt:
                elapsed = (now - check_in_dt).total_seconds()
                if elapsed < self.cooldown_seconds:
                    remaining = int(self.cooldown_seconds - elapsed)
                    mins = remaining // 60
                    secs = remaining % 60
                    return AttendanceResult(
                        status="cooldown",
                        message=f"Vui lòng chờ {mins} phút {secs} giây để Check-out",
                        check_in=check_in_dt
                    )
            
            # Perform Check-out
            self.attendance_repo.update_checkout(conn, today_log['id'], now)
            return AttendanceResult(
                status="checked_out",
                message=f"CHECK-OUT THÀNH CÔNG: {user.name.upper()}",
                check_in=check_in_dt,
                check_out=now
            )