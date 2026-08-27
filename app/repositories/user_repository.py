from typing import Optional, List, Dict, Any
from app.models.user import User

class UserRepository:
    def create(self, conn: Any, user: User) -> int:
        """Tạo user mới trong MySQL"""
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO users (name, position, face_encoding)
                VALUES (%s, %s, %s)
            """, (user.name, user.position, user.face_encoding))
            return cursor.lastrowid
    
    def get_by_id(self, conn: Any, user_id: int) -> Optional[User]:
        """Lấy user theo ID"""
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, name, position, face_encoding 
                FROM users 
                WHERE id = %s
            """, (user_id,))
            row = cursor.fetchone()
            if row:
                return User(
                    id=row['id'],
                    name=row['name'],
                    position=row['position'],
                    face_encoding=row['face_encoding']
                )
            return None
    
    def get_all(self, conn: Any) -> List[User]:
        """Lấy tất cả users"""
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, name, position, face_encoding 
                FROM users 
                ORDER BY id
            """)
            rows = cursor.fetchall()
            return [
                User(
                    id=row['id'],
                    name=row['name'],
                    position=row['position'],
                    face_encoding=row['face_encoding']
                )
                for row in rows
            ]
    
    def update(self, conn: Any, user: User) -> bool:
        """Cập nhật user"""
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE users 
                SET name = %s, position = %s, face_encoding = %s
                WHERE id = %s
            """, (user.name, user.position, user.face_encoding, user.id))
            return cursor.rowcount > 0
    
    def delete(self, conn: Any, user_id: int) -> bool:
        """Xóa user"""
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
            return cursor.rowcount > 0
    
    def get_all_with_encodings(self, conn: Any) -> Dict[int, User]:
        """Lấy tất cả users với face encoding đã decode"""
        users = self.get_all(conn)
        user_map = {}
        for user in users:
            if user.face_encoding:
                user_map[user.id] = user
        return user_map