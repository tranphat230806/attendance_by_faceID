import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Database MySQL Configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', '3306'))
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'phat2026@')  # Thay mật khẩu MySQL của bạn
    DB_NAME = os.getenv('DB_NAME', 'attendance')
    
    # Face Recognition
    # Lưu ý: Với dlib distance, ngưỡng 0.5-0.6 là phù hợp.
    FACE_RECOGNITION_THRESHOLD = float(os.getenv('FACE_RECOGNITION_THRESHOLD', '0.38'))
    LIVENESS_THRESHOLD = float(os.getenv('LIVENESS_THRESHOLD', '0.85'))
    
    # Attendance
    COOLDOWN_SECONDS = int(os.getenv('COOLDOWN_SECONDS', '300'))  # 5 minutes
    FACE_HOLD_DURATION = float(os.getenv('FACE_HOLD_DURATION', '5.0'))  # 5 seconds
    
    # Camera
    CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '640'))
    CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '480'))
    CAMERA_FPS = int(os.getenv('CAMERA_FPS', '30'))
    
    # API
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', '8000'))
    
    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here-change-in-production')
    ALGORITHM = os.getenv('ALGORITHM', 'HS256')
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '30'))