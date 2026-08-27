from fastapi import Request

def get_engine(request: Request):
    """Dependency để lấy AttendanceEngine từ app state"""
    return request.app.state.engine

def get_db(request: Request):
    """Dependency để lấy Database từ app state"""
    return request.app.state.db

def get_user_repo(request: Request):
    """Dependency để lấy UserRepository từ app state"""
    return request.app.state.user_repo