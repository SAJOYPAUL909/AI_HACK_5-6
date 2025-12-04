import os
from werkzeug.datastructures import FileStorage

def save_file(file: FileStorage, filename: str, upload_folder: str) -> str:
    """
    Save file to disk, return absolute path.
    """
    path = os.path.join(upload_folder, filename)
    file.save(path)
    return path
