import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads"))
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./forensics.db")
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200MB
    ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tiff"}
