# Attendance System - Face Recognition + Liveness Detection

Backend system for employee attendance tracking using face recognition and liveness detection.

## Features

- **Face Recognition**: Identify employees by facial encoding (128-d embedding)
- **Liveness Detection**: Prevent spoofing with eye blink and texture analysis
- **Check-in/Check-out**: Track attendance with proper state management
- **Missing Checkout Detection**: Automatically mark missed checkouts when new day starts
- **Database Isolation**: Prevent concurrent registration issues with unique constraints
- **Mock Services**: Full mock implementations for testing without real ML models
- **Error Handling**: Comprehensive error handling for all edge cases

## Architecture

```
app/
├── config.py              # Configuration (DB, thresholds)
├── db.py                  # Database setup
├── models.py              # SQLAlchemy models
├── schemas.py             # Pydantic schemas
├── main.py                # FastAPI application
│
├── models/                # (Optional) Additional models
├── repositories/          # Data access layer
│   ├── user_repository.py
│   └── attendance_repository.py
│
└── services/              # Business logic
    ├── liveness_service.py       # Liveness detection (real + mock)
    ├── face_service.py           # Face recognition service
    └── attendance_service.py     # Core attendance logic

tests/
└── test_console.py        # Console test suite (8 test cases)
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Database

#### Option A: Using MySQL

1. Create `.env` file from `.env.example`:
```bash
cp .env.example .env
```

2. Edit `.env` and set your MySQL credentials:
```
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=attendance
```

3. Ensure MySQL is running and create the database:
```bash
mysql -u your_user -p
> CREATE DATABASE attendance;
> EXIT;
```

#### Option B: Using SQLite (for testing)

Leave `.env` as default. System will automatically fall back to SQLite if MySQL is unavailable.

### 3. Initialize Database


## Running Tests

Execute the console test suite:

```bash
python tests/test_console.py
```

### Test Cases

1. **TEST 1**: Employee check-in (first time today)
2. **TEST 2**: Employee check-out same day
3. **TEST 3**: Duplicate check-out rejection (already checked out)
4. **TEST 4**: Missing checkout detection across days
5. **TEST 5**: Liveness failure rejection
6. **TEST 6**: Face recognition failure rejection
7. **TEST 7**: Liveness + Face recognition pass (full flow)
8. **TEST 8**: Concurrent check-in protection


## Core Logic

### Attendance Flow

```
Frame Input
    ↓
Liveness Detection (REQUIRED - must pass)
    ↓
Face Detection (extract embedding)
    ↓
Face Recognition (identify user)
    ↓
Check Previous Days (mark missing_checkout)
    ↓
Attendance Logic:
  - No record today → CREATE check_in
  - check_in exists, no check_out → UPDATE check_out
  - Already checked_out → REJECT (already_checked_out)
    ↓
Return Result
```

### Missing Checkout Detection

When employee check-in, system automatically detects if they forgot to checkout on previous days:

```
Yesterday:  08:00 check_in → NULL check_out → status=checked_in
Today:      08:02 check_in
            
Action: Mark yesterday as missing_checkout, create new check_in for today

Result:
Yesterday:  08:00 → NULL → missing_checkout
Today:      08:02 → NULL → checked_in
```

### Services

#### LivenessDetector (Interface)
- `predict(frame) -> Tuple[bool, float]`: Returns (is_alive, confidence_score)
- Implementations:
  - `RealLivenessDetector`: Eye blink + texture analysis (requires MediaPipe)
  - `MockLivenessDetector`: Mock for testing

#### FaceRecognitionService (Interface)
- `get_face_encoding(frame) -> np.ndarray`: Extract 128-d embedding
- `recognize_face(db, encoding) -> Tuple[Optional[int], float]`: Identify user
- Implementations:
  - `FaceRecognitionService`: Real recognition (requires face_recognition library)
  - `MockFaceRecognitionService`: Mock for testing

#### AttendanceService (Business Logic)
- `process_attendance(db, frame, date) -> Dict`: Main entry point
  - Validates liveness
  - Extracts face encoding
  - Recognizes user
  - Checks for missing checkout
  - Updates attendance status
  - Returns detailed result

## Configuration

See `.env.example` for all available options:

```
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=attendance
DB_TEST_NAME=attendance_test

FACE_RECOGNITION_THRESHOLD=0.45
LIVENESS_THRESHOLD=0.5
```

## Thresholds

- **FACE_RECOGNITION_THRESHOLD** (default: 0.45): Maximum face distance for match
  - Lower = stricter matching
  - Higher = more lenient
- **LIVENESS_THRESHOLD** (default: 0.5): Minimum confidence score for liveness
  - Prevents replay/spoofing attacks

## Error Handling

The system handles these errors gracefully:

- `liveness_failed`: Face is not live (replay attack, etc.)
- `face_not_detected`: No face found in frame
- `face_not_recognized`: Face detected but not in database
- `already_checked_out`: User already checked out today
- `invalid_user`: Database inconsistency
- Database connection errors

No exceptions are allowed to crash the service. All errors return proper error responses.

## Database Transactions

All attendance operations are transaction-safe:
- Unique constraint prevents duplicate check-in
- Foreign key ensures data integrity
- Status transitions are atomic

## Development Notes

### Adding Real ML Models

To use real models:

1. Replace `MockLivenessDetector` with `RealLivenessDetector` in services
2. Replace `MockFaceRecognitionService` with `FaceRecognitionService`
3. Adjust thresholds based on your models

### Using Different Database

The system supports any SQLAlchemy-compatible database. Change `DATABASE_URL` in `.env`:

- MySQL: `mysql+pymysql://user:pass@host/dbname`
- PostgreSQL: `postgresql://user:pass@host/dbname`
- SQLite: `sqlite:///./attendance.db`

## Future Enhancements

- [ ] Web UI for registration and attendance viewing
- [ ] Real-time dashboard with analytics
- [ ] Multi-camera support
- [ ] Advanced liveness detection (3D model, eye tracking)
- [ ] Attendance reports (CSV, PDF)
- [ ] API authentication and rate limiting
- [ ] Logging and audit trail
- [ ] Notification system (email, SMS)

## License

Proprietary
