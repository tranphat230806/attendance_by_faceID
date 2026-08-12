from pydantic import BaseModel
from typing import Optional

class FaceFrame(BaseModel):
    image: str                     # base64‑encoded JPEG from the browser
    action: Optional[str] = None  # optional "blink", "head_turn" etc. (used for liveness)

class RegisterFrame(BaseModel):
    image: str
    name: str
    position: str

class AttendanceResult(BaseModel):
    success: bool
    message: str
    name: Optional[str] = None
    position: Optional[str] = None
    check_in: Optional[str] = None
    check_out: Optional[str] = None

class RegistrationResult(BaseModel):
    success: bool
    message: str
    name: Optional[str] = None
    position: Optional[str] = None
