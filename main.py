import sys
import os
from pathlib import Path
import time
import threading
from fastapi import FastAPI
from datetime import datetime, timedelta, time as time_type
from contextlib import asynccontextmanager
from api.routes import users # Import router users
print("🔥 ĐANG CHẠY FILE MAIN.PY")
print("FILE:", __file__)

import cv2

print("CV2 PATH:", cv2.__file__)
print("CV2 VERSION:", cv2.__version__)
print("HAS IMDECODE:", hasattr(cv2, "imdecode"))

# Đăng ký router với prefix chuẩn /api/users
# 1. Khởi tạo app FastAPI TRƯỚC
app = FastAPI(title="Hệ thống điểm danh AI")

# 2. Đăng ký router SAU KHI đã có biến app
app.include_router(users.router, prefix="/api/users", tags=["Users"])
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import face_recognition

from app.config import Config
from app.database import Database
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.repositories.attendance_repository import AttendanceRepository
from app.services.liveness_service import RealLivenessDetector
from app.services.face_service import VectorFaceRecognizer
from app.services.attendance_service import AttendanceService

# Helper format time cho MySQL
def format_mysql_time(val) -> str:
    if val is None:
        return "--:--:--"
    if isinstance(val, timedelta):
        total_seconds = int(val.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    if isinstance(val, (datetime, time_type)):
        return val.strftime("%H:%M:%S")
    return str(val)

# ========== CORE SYSTEM ==========

class UltraFastCameraStream:
    def __init__(self, src=0):
        if sys.platform == "win32":
            self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.stream.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
                break
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()

class AsyncAttendanceEngine:
    def __init__(self, db: Database, user_repo: UserRepository, attendance_repo: AttendanceRepository):
        self.db = db
        self.user_repo = user_repo
        self.attendance_repo = attendance_repo
        self.liveness_detector = RealLivenessDetector(threshold=Config.LIVENESS_THRESHOLD)
        self.face_recognizer = VectorFaceRecognizer(user_repo=user_repo, threshold=Config.FACE_RECOGNITION_THRESHOLD)
        self.attendance_service = AttendanceService(
            db=db,
            user_repo=user_repo,
            attendance_repo=attendance_repo,
            liveness_detector=self.liveness_detector,
            face_recognizer=self.face_recognizer,
            cooldown_seconds=Config.COOLDOWN_SECONDS
        )
        with self.db.transaction() as conn:
            self.face_recognizer.reload_cache(conn)
        self.is_processing = False
        self.last_ai_time = 0
        self.ai_interval = 0.15
        self.prev_gray = None
        self.face_hold_tracker = {}
        self.ear_history_tracker = {}
        self.last_reported_sec = {}
        self.face_hold_duration = Config.FACE_HOLD_DURATION
        self.last_user_attendance = {}
        self.last_cooldown_warn = {}
        self.cooldown_seconds = Config.COOLDOWN_SECONDS
        self.latest_face_encodings = []
        self.lock = threading.Lock()

    def reload_users(self):
        with self.db.transaction() as conn:
            self.face_recognizer.reload_cache(conn)

    def reset_trackers(self):
        self.face_hold_tracker.clear()
        self.ear_history_tracker.clear()
        self.last_reported_sec.clear()

    def process_frame_sync(self, frame):
        """Xử lý frame đồng bộ cho API"""
        try:
            h, w, _ = frame.shape
            small_frame = cv2.resize(frame, (0, 0), fx=0.20, fy=0.20)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
            
            if not face_locations:
                return {"status": "error", "message": "Không phát hiện khuôn mặt"}
            
            top, right, bottom, left = face_locations[0]
            face_encoding = face_encodings[0]
            
            top *= 5
            right *= 5
            bottom *= 5
            left *= 5
            
            pad = 20
            crop_top = max(0, top - pad)
            crop_bottom = min(h, bottom + pad)
            crop_left = max(0, left - pad)
            crop_right = min(w, right + pad)
            crop_face = frame[crop_top:crop_bottom, crop_left:crop_right]
            
            avg_ear, texture_score = self.liveness_detector.get_eye_aspect_ratio(crop_face)
            
            if avg_ear is None:
                return {"status": "error", "message": "Phát hiện ảnh giả hoặc phản xạ màn hình"}
            
            user_id, face_score = self.face_recognizer.recognize(None, face_encoding)
            
            if not user_id:
                return {"status": "error", "message": "Khuôn mặt chưa đăng ký"}
            
            matched_user = self.face_recognizer.user_map.get(user_id)
            
            if not matched_user:
                return {"status": "error", "message": "Lỗi dữ liệu người dùng"}
            
            result = self.attendance_service.process_attendance(
                crop_face, 
                face_encoding, 
                now=datetime.now()
            )
            
            return {
                "status": result.status,
                "message": result.message,
                "user_id": matched_user.id,
                "user_name": matched_user.name,
                "position": matched_user.position,
                "score": float(face_score) if face_score else 0.0,
                "check_in": format_mysql_time(result.check_in) if result.check_in else None,
                "check_out": format_mysql_time(result.check_out) if result.check_out else None
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def process_frame_async(self, frame):
        now = time.time()
        if self.is_processing or frame is None or (now - self.last_ai_time < self.ai_interval):
            return
        self.is_processing = True
        self.last_ai_time = now
        t = threading.Thread(target=self._worker_process, args=(frame.copy(),), daemon=True)
        t.start()

    def _worker_process(self, frame):
        try:
            h, w, _ = frame.shape
            small_frame = cv2.resize(frame, (0, 0), fx=0.20, fy=0.20)
            gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            if self.prev_gray is not None and self.prev_gray.shape == gray_small.shape:
                frame_delta = cv2.absdiff(self.prev_gray, gray_small)
                motion_score = float(np.mean(frame_delta))
                self.prev_gray = gray_small
                if motion_score < 0.6 and not self.face_hold_tracker:
                    return
            self.prev_gray = gray_small
            
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
            
            with self.lock:
                self.latest_face_encodings = face_encodings
            
            if not face_locations:
                self.reset_trackers()
                return
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                top *= 5
                right *= 5
                bottom *= 5
                left *= 5
                
                pad = 20
                crop_top = max(0, top - pad)
                crop_bottom = min(h, bottom + pad)
                crop_left = max(0, left - pad)
                crop_right = min(w, right + pad)
                crop_face = frame[crop_top:crop_bottom, crop_left:crop_right]
                
                avg_ear, texture_score = self.liveness_detector.get_eye_aspect_ratio(crop_face)
                user_id, face_score = self.face_recognizer.recognize(None, face_encoding)
                matched_user = self.face_recognizer.user_map.get(user_id) if user_id else None
                
                now_ts = time.time()
                time_str = datetime.now().strftime("%H:%M:%S")
                
                if avg_ear is None:
                    print(f"\n[{time_str}] ⚠️ CẢNH BÁO: Phát hiện PHẢN XẠ MÀN HÌNH ĐIỆN THOẠI / Ảnh chụp không hợp lệ!")
                    self.reset_trackers()
                    continue
                
                if matched_user:
                    uid = matched_user.id
                    last_att = self.last_user_attendance.get(uid, 0)
                    if now_ts - last_att < self.cooldown_seconds:
                        rem_secs = int(self.cooldown_seconds - (now_ts - last_att))
                        mins = rem_secs // 60
                        secs = rem_secs % 60
                        if now_ts - self.last_cooldown_warn.get(uid, 0) > 5.0:
                            self.last_cooldown_warn[uid] = now_ts
                            print(f"\n[{time_str}] ⏳ NV {matched_user.name.upper()} vừa điểm danh. Vui lòng chờ {mins} phút {secs} giây để Check-out!")
                        continue
                    
                    if uid not in self.ear_history_tracker:
                        self.ear_history_tracker[uid] = []
                    self.ear_history_tracker[uid].append(avg_ear)
                    if len(self.ear_history_tracker[uid]) > 30:
                        self.ear_history_tracker[uid].pop(0)
                    
                    if uid not in self.face_hold_tracker:
                        self.face_hold_tracker[uid] = now_ts
                        self.last_reported_sec[uid] = 0
                        print(f"\n[{time_str}] 🔍 Đã phát hiện NV: {matched_user.name.upper()} - Vui lòng GIỮ YÊN 5 giây...")
                    else:
                        elapsed = now_ts - self.face_hold_tracker[uid]
                        curr_sec = int(elapsed)
                        
                        if curr_sec > self.last_reported_sec.get(uid, 0) and curr_sec <= 5:
                            self.last_reported_sec[uid] = curr_sec
                            print(f"[{time_str}] ⏳ Kiểm tra chớp mắt & liveness NV: {matched_user.name.upper()} ({curr_sec}/5 giây)...")
                        
                        if elapsed >= self.face_hold_duration:
                            is_dynamic_live, live_score = self.liveness_detector.predict_dynamic(
                                self.ear_history_tracker.get(uid, []),
                                texture_score
                            )
                            
                            if not is_dynamic_live:
                                print(f"\n[{time_str}] ❌ TỪ CHỐI ĐIỂM DANH: Phát hiện ẢNH CHỤP ĐIỆN THOẠI! (Score: {live_score:.2f})\n")
                                self.reset_trackers()
                                continue
                            
                            self.last_user_attendance[uid] = now_ts
                            self.reset_trackers()
                            
                            res = self.attendance_service.process_attendance(crop_face, face_encoding, now=datetime.now())
                            
                            if res.status == "checked_in":
                                print(f"\n[{time_str}] 🎉 CHECK-IN THÀNH CÔNG: {matched_user.name.upper()} ({matched_user.position})\n")
                            elif res.status == "checked_out":
                                print(f"\n[{time_str}] 🟡 CHECK-OUT THÀNH CÔNG: {matched_user.name.upper()} ({matched_user.position})\n")
        except Exception:
            pass
        finally:
            self.is_processing = False

    def get_encodings(self):
        with self.lock:
            return list(self.latest_face_encodings)

# ========== CLI FUNCTIONS ==========

def start_camera_attendance(engine: AsyncAttendanceEngine):
    print("\n" + "="*60)
    print(" 🎥 ĐANG MỞ CAMERA ĐIỂM DANH (CHỐNG ẢNH ĐIỆN THOẠI | BẮT CHỚP MẮT)")
    print("==================================================")
    print(" • Nhân viên đứng trước Camera và GIỮ YÊN 5 GIÂY để xác nhận điểm danh.")
    print(" • Nhấn phím 'Q' hoặc 'ESC' trên cửa sổ Camera để QUAY LẠI MENU.")
    print("==================================================\n")
    
    webcam = UltraFastCameraStream(src=0).start()
    time.sleep(0.2)
    
    if not webcam.grabbed:
        print("❌ Không thể mở Webcam! Vui lòng kiểm tra thiết bị camera.")
        webcam.stop()
        return
    
    while True:
        ret, frame = webcam.read()
        if not ret or frame is None:
            break
        frame = cv2.flip(frame, 1)
        engine.process_frame_async(frame)
        cv2.imshow("DIEM DANH KHUON MAT (Nhan Q de Thoat)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("\n⏹ Đã dừng Camera. Quay lại Menu chính.")
            break
    
    webcam.stop()
    cv2.destroyAllWindows()

def register_user_from_camera(db: Database, engine: AsyncAttendanceEngine, user_repo: UserRepository):
    print("\n--------------------------------------------------")
    print("--- 2. ĐĂNG KÝ NHÂN VIÊN MỚI TỪ CAMERA ---")
    print("--------------------------------------------------")
    print(" 1. Nhìn vào màn hình Camera và căn chỉnh vị trí khuôn mặt.")
    print(" 2. Nhấn phím [SPACE] (phím Cách) hoặc [S] để BẮT ĐẦU ĐĂNG KÝ.")
    print(" 3. Nhấn [Q] hoặc [ESC] để HỦY BỎ đăng ký.")
    print("--------------------------------------------------\n")
    
    webcam = UltraFastCameraStream(src=0).start()
    time.sleep(0.3)
    
    if not webcam.grabbed:
        print("❌ Không thể mở Webcam!")
        webcam.stop()
        return
    
    captured_encoding = None
    
    while True:
        ret, frame = webcam.read()
        if not ret or frame is None:
            break
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        cv2.putText(display_frame, "NHAN [SPACE] DE BAT DAU DANG KY | [Q] THOAT", (15, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 0), 1)
        cv2.imshow("CUA SO DANG KY NHAN VIEN - Nhan [SPACE] de Chup", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key in (32, ord('s'), ord('S')):
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb_small, model="hog")
            encs = face_recognition.face_encodings(rgb_small, locs)
            if not encs:
                print("\n❌ Không phát hiện khuôn mặt nào! Vui lòng nhìn thẳng vào Camera và thử lại.")
                continue
            captured_encoding = encs[0]
            print("\n📸 Đã chụp và trích xuất thành công khuôn mặt!")
            break
        elif key in (ord('q'), ord('Q'), 27):
            print("\n⏹ Đã hủy đăng ký nhân viên.")
            break
    
    webcam.stop()
    cv2.destroyAllWindows()
    
    if captured_encoding is None:
        return
    
    matched_id, score = engine.face_recognizer.recognize(None, captured_encoding)
    if matched_id:
        existing_user = engine.face_recognizer.user_map.get(matched_id)
        user_name = existing_user.name if existing_user else f"ID {matched_id}"
        print(f"\n❌ KHÔNG THỂ ĐĂNG KÝ TRÙNG LẶP!")
        print(f"Khuôn mặt này ĐÃ ĐƯỢC ĐĂNG KÝ trước đó bởi: {user_name} (ID: {matched_id}) | Độ khớp: {score*100:.1f}%\n")
        return
    
    name = input("\nNhập họ và tên nhân viên: ").strip()
    if not name:
        print("❌ Tên không được để trống! Đăng ký bị hủy.")
        return
    position = input("Nhập chức vụ (mặc định: Nhân viên): ").strip() or "Nhân viên"
    
    new_user = User(
        id=None,
        name=name,
        position=position,
        face_encoding=User.encode_embedding(captured_encoding)
    )
    
    with db.transaction() as conn:
        new_id = user_repo.create(conn, new_user)
    
    engine.reload_users()
    print(f"\n🎉 ĐĂNG KÝ THÀNH CÔNG! ID: {new_id} | Tên: {name} | Chức vụ: {position}\n")

def list_users_and_logs(db: Database):
    print("\n==================================================")
    print(" 📋 DANH SÁCH NHÂN VIÊN & LỊCH SỬ ĐIỂM DANH HÔM NAY")
    print("==================================================")

    with db.transaction() as conn:
        with conn.cursor() as cursor:
            # Danh sách nhân viên
            cursor.execute("""
                SELECT id, name, position, created_at
                FROM users;
            """)
            users = cursor.fetchall()

            # Lịch sử điểm danh hôm nay (Cú pháp CURDATE() chuẩn MySQL)
            cursor.execute("""
                SELECT
                    a.id,
                    u.name,
                    a.date,
                    a.check_in,
                    a.check_out,
                    a.status
                FROM attendance_logs a
                JOIN users u ON a.user_id = u.id
                WHERE a.date = CURDATE()
                ORDER BY a.id DESC;
            """)
            logs = cursor.fetchall()

    print("\n--- DANH SÁCH NHÂN VIÊN ĐÃ ĐĂNG KÝ ---")
    if not users:
        print("Chưa có nhân viên nào.")
    else:
        print(f"{'ID':<6} | {'Tên nhân viên':<25} | {'Chức vụ':<20}")
        print("-" * 55)
        for u in users:
            print(f"{u['id']:<6} | {u['name']:<25} | {u['position']:<20}")

    print("\n--- NHẬT KÝ ĐIỂM DANH HÔM NAY ---")
    if not logs:
        print("Chưa có nhật ký điểm danh hôm nay.")
    else:
        print(f"{'ID':<6} | {'Tên NV':<22} | {'Check-in':<10} | {'Check-out':<10} | {'Trạng thái':<16}")
        print("-" * 75)
        for r in logs:
            c_in = format_mysql_time(r["check_in"])
            c_out = format_mysql_time(r["check_out"])
            print(f"{r['id']:<6} | {r['name']:<22} | {c_in:<10} | {c_out:<10} | {r['status']:<16}")
    print()

# ========== FASTAPI SETUP WITH LIFESPAN ==========

# Khởi tạo đối tượng toàn cục
db_instance = Database(Config.DB_NAME)
user_repo_instance = UserRepository()
attendance_repo_instance = AttendanceRepository()
engine_instance = AsyncAttendanceEngine(db_instance, user_repo_instance, attendance_repo_instance)

# ========== FASTAPI SETUP WITH LIFESPAN ==========

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import attendance, users
import uvicorn

# 1. Khai báo các đối tượng toàn cục
db_instance = Database(Config.DB_NAME)
user_repo_instance = UserRepository()
attendance_repo_instance = AttendanceRepository()
engine_instance = AsyncAttendanceEngine(db_instance, user_repo_instance, attendance_repo_instance)

# 2. Định nghĩa Lifespan (sau khi đã import FastAPI)
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = db_instance
    app.state.user_repo = user_repo_instance
    app.state.attendance_repo = attendance_repo_instance
    app.state.engine = engine_instance
    yield

# 3. Khởi tạo FastAPI App với lifespan
app = FastAPI(
    title="Hệ Thống Điểm Danh Khuôn Mặt AI",
    description="API điểm danh thông minh với AI",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(attendance.router, prefix="/api/attendance", tags=["Attendance"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])

@app.get("/")
async def root():
    return {
        "message": "🎯 Hệ Thống Điểm Danh Khuôn Mặt AI",
        "version": "2.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "attendance": {
                "process": "/api/attendance/process",
                "status": "/api/attendance/check-status/{user_id}",
                "today": "/api/attendance/today"
            },
            "users": {
                "register": "/api/users/register",
                "list": "/api/users/list",
                "delete": "/api/users/{user_id}"
            }
        }
    }
# ========== MAIN MENU ==========

def main_cli_menu():
    while True:
        print("\n" + "="*60)
        print("   HỆ THỐNG ĐIỂM DANH KHUÔN MẶT - CONSOLE MENU")
        print("="*60)
        print(" 1. Mở Camera điểm danh tự động")
        print(" 2. Đăng ký nhân viên mới (từ Camera)")
        print(" 3. Xem danh sách nhân viên & Lịch sử điểm danh")
        print(" 4. 🚀 Chạy API Server (Swagger: http://localhost:8000/docs)")
        print(" 0. Thoát")
        print("-"*60)
        choice = input("Nhập lựa chọn của bạn (0-4): ").strip()

        if choice == "1":
            start_camera_attendance(engine_instance)
        elif choice == "2":
            register_user_from_camera(db_instance, engine_instance, user_repo_instance)
        elif choice == "3":
            list_users_and_logs(db_instance)
        elif choice == "4":
            print("\n🚀 Đang khởi động API Server...")
            print("📚 Swagger UI: http://localhost:8000/docs")
            print("📖 ReDoc: http://localhost:8000/redoc")
            print("Press Ctrl+C để dừng server\n")
            uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)
        elif choice == "0":
            print("\nCảm ơn bạn đã sử dụng hệ thống. Tạm biệt!")
            break
        else:
            print("\nLựa chọn không hợp lệ, vui lòng chọn lại!\n")

if __name__ == "__main__":
    main_cli_menu()