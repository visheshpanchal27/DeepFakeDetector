"""
Configuration settings for the Enhanced DeepFake Detector
"""

import os
from pathlib import Path

class Config:
    """Configuration class for the detector"""
    
    # File validation settings
    ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
    MAX_FILE_SIZE_MB = 100
    MIN_IMAGE_SIZE = (32, 32)
    MAX_IMAGE_SIZE = (4096, 4096)
    
    # Detection thresholds
    AUTHENTICITY_THRESHOLDS = {
        'AUTHENTIC_HUMAN': 0.80,
        'HUMAN_ENHANCED': 0.60,
        'SUSPICIOUS': 0.40,
        'AI_GENERATED': 0.0
    }
    
    # Analysis weights
    ANALYSIS_WEIGHTS = {
        'texture_analysis': 0.25,
        'frequency_domain': 0.20,
        'facial_consistency': 0.15,
        'ai_signatures': 0.30,
        'compression_artifacts': 0.10
    }
    
    # Security settings
    ENABLE_PATH_VALIDATION = True
    ENABLE_FILE_SIZE_CHECK = True
    SANITIZE_FILENAMES = True
    
    # Logging configuration
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/deepfake_detector.log'
    MAX_LOG_SIZE_MB = 10
    
    # Report settings
    REPORTS_DIR = 'AI_Detection_Reports'
    GENERATE_HTML_REPORTS = True
    GENERATE_JSON_DATA = True
    GENERATE_CSV_EXPORT = True
    
    # Performance settings
    MAX_BATCH_SIZE = 1000
    ENABLE_MULTIPROCESSING = False
    MAX_WORKERS = 4
    
    @classmethod
    def get_reports_dir(cls) -> Path:
        """Get reports directory path"""
        reports_path = Path(cls.REPORTS_DIR)
        reports_path.mkdir(exist_ok=True)
        return reports_path
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure required directories exist when needed with security validation"""
        try:
            # Validate directory names to prevent path traversal
            reports_dir = os.path.basename(cls.REPORTS_DIR)
            if reports_dir and '..' not in reports_dir:
                Path(reports_dir).mkdir(exist_ok=True)
            
            results_dir = 'Analysis_Results'
            if '..' not in results_dir:
                Path(results_dir).mkdir(exist_ok=True)
        except (OSError, ValueError):
            # Silently ignore directory creation errors
            pass
    
    @classmethod
    def validate_file_size(cls, file_path: str) -> bool:
        """Validate file size with security checks"""
        if not cls.ENABLE_FILE_SIZE_CHECK:
            return True
        
        try:
            # Resolve path to prevent traversal attacks
            resolved_path = Path(file_path).resolve()
            
            # Allow files from anywhere, just prevent dangerous paths
            if '..' in str(resolved_path):
                return False
            
            if not resolved_path.exists() or not resolved_path.is_file():
                return False
                
            file_size = resolved_path.stat().st_size
            max_size = cls.MAX_FILE_SIZE_MB * 1024 * 1024
            return file_size <= max_size
        except (OSError, ValueError):
            return False
    
    @classmethod
    def get_classification(cls, score: float) -> str:
        """Get classification based on score"""
        if score >= cls.AUTHENTICITY_THRESHOLDS['AUTHENTIC_HUMAN']:
            return 'AUTHENTIC_HUMAN'
        elif score >= cls.AUTHENTICITY_THRESHOLDS['HUMAN_ENHANCED']:
            return 'HUMAN_ENHANCED'
        elif score >= cls.AUTHENTICITY_THRESHOLDS['SUSPICIOUS']:
            return 'SUSPICIOUS'
        else:
            return 'AI_GENERATED'