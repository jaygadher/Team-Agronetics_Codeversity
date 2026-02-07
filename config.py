import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-for-development-only')
    DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
    
    # File Uploads
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # Model
    MODEL_PATH = os.getenv('MODEL_PATH', 'plant_disease_model.pth')
    
    # Ensure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Detection results
    DETECTION_RESULTS = 'detection_results'
    os.makedirs(DETECTION_RESULTS, exist_ok=True)

# Create an instance of the config
config = Config()
