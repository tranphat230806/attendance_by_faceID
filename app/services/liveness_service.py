import cv2
import numpy as np
import face_recognition
from typing import Tuple, Optional

class RealLivenessDetector:
    """Phát hiện liveness bằng EAR và texture analysis"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.eye_landmarks = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    
    def get_eye_aspect_ratio(self, face_image: np.ndarray) -> Tuple[Optional[float], float]:
        """
        Tính EAR và texture score
        Returns: (avg_ear, texture_score)
        """
        try:
            # Phát hiện landmarks
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_landmarks = face_recognition.face_landmarks(rgb_face)
            
            if not face_landmarks:
                return None, 0.0
            
            landmarks = face_landmarks[0]
            
            # Lấy eye landmarks
            left_eye = landmarks.get('left_eye', [])
            right_eye = landmarks.get('right_eye', [])
            
            if len(left_eye) < 6 or len(right_eye) < 6:
                return None, 0.0
            
            # Tính EAR cho mắt trái
            left_ear = self._calculate_ear(left_eye)
            right_ear = self._calculate_ear(right_eye)
            
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Tính texture score (phát hiện phản xạ màn hình)
            texture_score = self._calculate_texture_score(face_image)
            
            return avg_ear, texture_score
            
        except Exception as e:
            return None, 0.0
    
    def _calculate_ear(self, eye_points) -> float:
        """Tính Eye Aspect Ratio"""
        # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        p1, p2, p3, p4, p5, p6 = eye_points
        
        # Vertical distances
        v1 = np.linalg.norm(np.array(p2) - np.array(p6))
        v2 = np.linalg.norm(np.array(p3) - np.array(p5))
        
        # Horizontal distance
        h = np.linalg.norm(np.array(p1) - np.array(p4))
        
        if h == 0:
            return 0.0
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def _calculate_texture_score(self, face_image: np.ndarray) -> float:
        """Tính texture score để phát hiện ảnh giả"""
        try:
            # Chuyển sang grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Tính Laplacian variance (đo độ sắc nét)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Chuẩn hóa về [0, 1]
            texture_score = min(1.0, laplacian_var / 1000.0)
            
            return texture_score
            
        except Exception:
            return 0.0
    
    def predict_dynamic(self, ear_history: list, texture_score: float) -> Tuple[bool, float]:
        """
        Dự đoán liveness dựa trên lịch sử EAR và texture
        Returns: (is_live, confidence_score)
        """
        # Kiểm tra có đủ dữ liệu không
        if len(ear_history) < 5:
            return False, 0.0
        
        # Tính variance của EAR (blink detection)
        ear_array = np.array(ear_history[-30:])
        ear_std = np.std(ear_array)
        
        # EAR trung bình
        ear_mean = np.mean(ear_array)
        
        # Điểm liveness
        live_score = 0.0
        
        # 1. EAR variance (chớp mắt)
        if ear_std > 0.02:  # Có sự thay đổi EAR (chớp mắt)
            live_score += 0.5
        elif ear_std > 0.01:
            live_score += 0.3
        
        # 2. Texture score
        if texture_score > 0.3:
            live_score += 0.3
        else:
            live_score += 0.1
        
        # 3. EAR mean (mắt mở)
        if 0.2 < ear_mean < 0.4:
            live_score += 0.2
        
        # Quyết định
        is_live = live_score >= self.threshold
        
        return is_live, live_score