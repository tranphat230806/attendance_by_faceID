import json
import numpy as np

class User:
    def __init__(self, id=None, name=None, position=None, face_encoding=None):
        self.id = id
        self.name = name
        self.position = position
        self.face_encoding = face_encoding  # Lưu dạng list hoặc None
    
    @staticmethod
    def encode_embedding(embedding):
        """Chuyển numpy array thành string JSON để lưu vào DB"""
        if embedding is None:
            return None
        if isinstance(embedding, np.ndarray):
            return json.dumps(embedding.tolist())
        return json.dumps(embedding)
    
    @staticmethod
    def decode_embedding(embedding_str):
        """Chuyển string JSON từ DB thành numpy array"""
        if embedding_str is None:
            return None
        try:
            data = json.loads(embedding_str)
            return np.array(data)
        except:
            return None
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'position': self.position,
            'face_encoding': self.face_encoding
        }