import numpy as np
import face_recognition
from typing import Optional, Tuple, Dict, Any
from app.models.user import User
from app.repositories.user_repository import UserRepository

class VectorFaceRecognizer:
    """Nhận diện khuôn mặt bằng vector embeddings"""
    
    def __init__(self, user_repo: UserRepository, threshold: float = 0.5):
        self.user_repo = user_repo
        # Threshold khoảng cách (Euclidean Distance): Càng nhỏ càng khắt khe. Chuẩn 0.5 - 0.6
        self.threshold = threshold 
        self.user_map: Dict[int, User] = {}
        self.encodings: Dict[int, np.ndarray] = {}
    
    def reload_cache(self, conn: Any):
        """Tải lại cache từ database"""
        users = self.user_repo.get_all(conn)
        self.user_map.clear()
        self.encodings.clear()
        
        for user in users:
            if user.face_encoding:
                encoding = User.decode_embedding(user.face_encoding)
                if encoding is not None:
                    self.user_map[user.id] = user
                    self.encodings[user.id] = encoding
        
        print(f"✅ Đã tải {len(self.user_map)} users vào cache")
    
    def recognize(self, conn: Any, face_encoding: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """
        Nhận diện khuôn mặt sử dụng Vectorization tối ưu hiệu năng
        Returns: (user_id, confidence_score)
        """
        if not self.encodings or face_encoding is None:
            return None, None
        
        # Lấy danh sách ID và danh sách Vector Encodings
        user_ids = list(self.encodings.keys())
        known_encodings_list = list(self.encodings.values())
        
        # Tính khoảng cách đến TẤT CẢ user cùng 1 lúc (Numpy Vectorized)
        distances = face_recognition.face_distance(known_encodings_list, face_encoding)
        
        # Tìm khoảng cách nhỏ nhất (người giống nhất)
        best_match_index = np.argmin(distances)
        min_distance = distances[best_match_index]
        
        # Kiểm tra ngưỡng (Khoảng cách phải NHỎ HƠN hoặc BẰNG threshold)
        if min_distance <= self.threshold:
            best_match_id = user_ids[best_match_index]
            # Độ tin cậy = (1 - distance)
            confidence_score = float(1.0 - min_distance)
            return best_match_id, confidence_score
        
        return None, None
    
    def add_user(self, conn: Any, user: User):
        """Thêm user mới vào cache lập tức không cần query lại DB"""
        if user.id and user.face_encoding:
            encoding = User.decode_embedding(user.face_encoding)
            if encoding is not None:
                self.user_map[user.id] = user
                self.encodings[user.id] = encoding
    
    def remove_user(self, user_id: int):
        """Xóa user khỏi cache"""
        if user_id in self.user_map:
            del self.user_map[user_id]
        if user_id in self.encodings:
            del self.encodings[user_id]