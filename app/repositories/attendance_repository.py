from typing import Optional, List, Dict, Any
from datetime import datetime

class AttendanceRepository:
    def create_log(self, conn: Any, user_id: int, check_in: datetime) -> int:
        """Tạo log điểm danh mới trong MySQL"""
        with conn.cursor() as cursor:
            # Thêm date = CURDATE() vì schema MySQL yêu cầu date NOT NULL
            cursor.execute("""
                INSERT INTO attendance_logs (user_id, date, check_in, status)
                VALUES (%s, CURDATE(), %s, 'checked_in')
            """, (user_id, check_in))
            return cursor.lastrowid
    
    def update_checkout(self, conn: Any, log_id: int, check_out: datetime) -> bool:
        """Cập nhật check-out"""
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE attendance_logs 
                SET check_out = %s, status = 'checked_out'
                WHERE id = %s
            """, (check_out, log_id))
            return cursor.rowcount > 0
    
    def get_today_log(self, conn: Any, user_id: int) -> Optional[Dict]:
        """Lấy log điểm danh hôm nay của user"""
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, user_id, date, check_in, check_out, status
                FROM attendance_logs
                WHERE user_id = %s AND date = CURDATE()
                ORDER BY id DESC LIMIT 1
            """, (user_id,))
            return cursor.fetchone()
    
    def get_today_logs(self, conn: Any) -> List[Dict]:
        """Lấy tất cả log điểm danh hôm nay"""
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT a.id, u.name, u.position, a.date, a.check_in, a.check_out, a.status
                FROM attendance_logs a
                JOIN users u ON a.user_id = u.id
                WHERE a.date = CURDATE()
                ORDER BY a.id DESC
            """)
            return cursor.fetchall()
    
    def has_checked_in_today(self, conn: Any, user_id: int) -> bool:
        """Kiểm tra user đã check-in hôm nay chưa"""
        log = self.get_today_log(conn, user_id)
        return log is not None
    
    def get_last_checkout(self, conn: Any, user_id: int) -> Optional[Dict]:
        """Lấy log check-out gần nhất"""
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, check_out, status
                FROM attendance_logs
                WHERE user_id = %s AND status = 'checked_out'
                ORDER BY check_out DESC LIMIT 1
            """, (user_id,))
            return cursor.fetchone()