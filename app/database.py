from contextlib import contextmanager
import pymysql
from pymysql.cursors import DictCursor
from app.config import Config

class Database:
    def __init__(
        self, 
        db_name: str = Config.DB_NAME,
        host: str = Config.DB_HOST,
        user: str = Config.DB_USER,
        password: str = Config.DB_PASSWORD,
        port: int = Config.DB_PORT
    ):
        # Tự động cắt bỏ đuôi .db nếu lỡ truyền vào
        if db_name.endswith(".db"):
            db_name = db_name.replace(".db", "")

        self.db_name = db_name
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self._init_db()

    def get_raw_connection(self):
        """Tạo kết nối tới MySQL"""
        return pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.db_name,
            port=self.port,
            cursorclass=DictCursor,
            autocommit=False,
            ssl={"fake_flag_to_enable_tls": True} if False else None,
            auth_plugin_map={}
        )

    def _init_db(self):
        """Khởi tạo các bảng nếu chưa có"""
        with self.transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        position VARCHAR(100) DEFAULT 'Nhân viên',
                        face_encoding LONGTEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS attendance_logs (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id INT NOT NULL,
                        date DATE NOT NULL,
                        check_in DATETIME NULL,
                        check_out DATETIME NULL,
                        status ENUM('checked_in', 'checked_out') NOT NULL DEFAULT 'checked_in',
                        check_in_liveness_score FLOAT NULL,
                        check_out_liveness_score FLOAT NULL,
                        check_in_face_score FLOAT NULL,
                        check_out_face_score FLOAT NULL,
                        CONSTRAINT fk_attendance_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                        UNIQUE KEY uq_user_date (user_id, date)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

    @contextmanager
    def get_connection(self):
        conn = self.get_raw_connection()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self):
        conn = self.get_raw_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()