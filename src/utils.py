"""
Utility functions for the Enhanced DeepFake Detector
"""

import cv2
import numpy as np
import os
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime

def setup_logging(log_level: str = 'INFO', log_file: str = 'detector.log') -> logging.Logger:
    """Setup logging configuration with security validation"""
    # Validate log level
    valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    if log_level.upper() not in valid_levels:
        log_level = 'INFO'
    
    # Sanitize log file path
    log_file = os.path.basename(log_file)  # Prevent path traversal
    if not log_file.endswith('.log'):
        log_file = 'detector.log'
    
    logger = logging.getLogger('deepfake_detector')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler with safe path
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
    except (OSError, IOError):
        # Fallback to console only if file handler fails
        file_handler = None
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    if file_handler:
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of file with security validation"""
    try:
        # Validate and sanitize file path
        resolved_path = Path(file_path).resolve()
        
        # Allow files from anywhere, just prevent dangerous paths
        if '..' in str(resolved_path):
            logging.warning(f"Suspicious path blocked: {file_path}")
            return ""
        
        if not resolved_path.exists() or not resolved_path.is_file():
            return ""
        
        # Check file size limit (100MB)
        if resolved_path.stat().st_size > 100 * 1024 * 1024:
            logging.warning(f"File too large: {file_path}")
            return ""
        
        hash_sha256 = hashlib.sha256()
        with open(resolved_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except (OSError, IOError, ValueError) as e:
        logging.error(f"Error calculating hash for {file_path}: {e}")
        return ""

def validate_image_format(image_path: str) -> bool:
    """Validate if file is a valid image with security checks"""
    try:
        # Validate and sanitize path
        resolved_path = Path(image_path).resolve()
        
        # Allow files from anywhere, just prevent dangerous paths
        if '..' in str(resolved_path):
            logging.warning(f"Suspicious path blocked: {image_path}")
            return False
        
        if not resolved_path.exists() or not resolved_path.is_file():
            return False
        
        # Check file extension
        allowed_extensions = {
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'
        }
        if resolved_path.suffix.lower() not in allowed_extensions:
            return False
        
        image = cv2.imread(str(resolved_path))
        return image is not None
    except Exception as e:
        logging.error(f"Error validating image {image_path}: {e}")
        return False

def resize_image_if_needed(image: np.ndarray, max_size: Tuple[int, int] = (2048, 2048)) -> np.ndarray:
    """Resize image if it exceeds maximum dimensions"""
    h, w = image.shape[:2]
    max_h, max_w = max_size
    
    if h > max_h or w > max_w:
        # Calculate scaling factor
        scale = min(max_h / h, max_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image

def extract_metadata(file_path: str) -> Dict:
    """Extract basic file metadata with security validation"""
    try:
        # Validate and sanitize path
        resolved_path = Path(file_path).resolve()
        
        # Allow files from anywhere, just prevent dangerous paths
        if '..' in str(resolved_path):
            logging.warning(f"Suspicious path blocked: {file_path}")
            return {}
        
        if not resolved_path.exists():
            return {}
        
        stat = resolved_path.stat()
        return {
            'file_size': stat.st_size,
            'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'file_hash': calculate_file_hash(str(resolved_path))
        }
    except Exception as e:
        logging.error(f"Error extracting metadata for {file_path}: {e}")
        return {}

def create_thumbnail(image: np.ndarray, size: Tuple[int, int] = (150, 150)) -> np.ndarray:
    """Create thumbnail of image"""
    try:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    except Exception:
        # Return original image if resize fails
        return image

def save_analysis_cache(results: Dict, cache_file: str = 'analysis_cache.json') -> None:
    """Save analysis results to cache with security validation"""
    try:
        # Sanitize cache file name
        cache_file = os.path.basename(cache_file)
        if not cache_file.endswith('.json'):
            cache_file = 'analysis_cache.json'
        
        cache_data = {}
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        
        # Add new results
        file_hash = results.get('file_hash', '')
        if file_hash and isinstance(file_hash, str) and len(file_hash) == 64:  # SHA-256 length
            cache_data[file_hash] = {
                'timestamp': datetime.now().isoformat(),
                'results': results
            }
        
        # Limit cache size
        if len(cache_data) > 1000:
            # Keep only the 500 most recent entries
            sorted_items = sorted(
                cache_data.items(), 
                key=lambda x: x[1]['timestamp'], 
                reverse=True
            )
            cache_data = dict(sorted_items[:500])
        
        # Save updated cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
            
    except Exception as e:
        logging.error(f"Error saving cache: {e}")

def load_analysis_cache(file_hash: str, cache_file: str = 'analysis_cache.json') -> Optional[Dict]:
    """Load analysis results from cache with validation"""
    try:
        # Validate file hash format
        if not isinstance(file_hash, str) or len(file_hash) != 64:
            return None
        
        # Sanitize cache file name
        cache_file = os.path.basename(cache_file)
        if not cache_file.endswith('.json'):
            cache_file = 'analysis_cache.json'
        
        if not os.path.exists(cache_file):
            return None
            
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        if file_hash in cache_data:
            cached_result = cache_data[file_hash]
            # Check if cache is recent (within 24 hours)
            try:
                cached_time = datetime.fromisoformat(cached_result['timestamp'])
                if (datetime.now() - cached_time).total_seconds() < 86400:
                    return cached_result['results']
            except (ValueError, KeyError):
                # Invalid timestamp format, ignore cached result
                pass
        
        return None
        
    except Exception as e:
        logging.error(f"Error loading cache: {e}")
        return None

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f}{size_names[i]}"

def get_supported_files(directory: str, extensions: set) -> List[str]:
    """Get list of supported files in directory with security validation"""
    supported_files = []
    
    try:
        # Validate directory path
        dir_path = Path(directory).resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            return supported_files
        
        # Limit directory traversal depth
        max_depth = 3
        for root, dirs, files in os.walk(dir_path):
            # Check depth limit
            depth = len(Path(root).relative_to(dir_path).parts)
            if depth > max_depth:
                dirs.clear()  # Don't recurse deeper
                continue
                
            for file in files:
                if Path(file).suffix.lower() in extensions:
                    file_path = Path(root) / file
                    supported_files.append(str(file_path))
                    
                    # Limit number of files to prevent memory issues
                    if len(supported_files) > 1000:
                        return supported_files
                        
    except (OSError, ValueError) as e:
        logging.error(f"Error scanning directory {directory}: {e}")
    
    return supported_files

def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """Create a simple progress bar"""
    if total == 0:
        return "[" + "=" * width + "]"
    
    progress = current / total
    filled = int(width * progress)
    bar = "=" * filled + "-" * (width - filled)
    percentage = progress * 100
    
    return f"[{bar}] {percentage:.1f}% ({current}/{total})"

def export_to_csv(data: List[Dict], output_file: str) -> bool:
    """Export analysis results to CSV with security validation"""
    try:
        import csv  # Import here to avoid dependency issues
        
        if not data or not isinstance(data, list):
            return False
        
        # Sanitize output file name
        output_file = os.path.basename(output_file)
        if not output_file.endswith('.csv'):
            output_file = 'analysis_results.csv'
        
        # Get all unique keys from all dictionaries
        fieldnames = set()
        for item in data:
            if isinstance(item, dict):
                fieldnames.update(item.keys())
        fieldnames = sorted(list(fieldnames))
        
        if not fieldnames:
            return False
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Sanitize data before writing
            for item in data:
                if isinstance(item, dict):
                    sanitized_item = {}
                    for key, value in item.items():
                        # Convert to string and sanitize
                        sanitized_item[key] = str(value).replace('\n', ' ').replace('\r', '')
                    writer.writerow(sanitized_item)
        
        return True
        
    except Exception as e:
        logging.error(f"Error exporting to CSV: {e}")
        return False

def clean_old_logs(log_dir: str = ".", max_age_days: int = 30) -> None:
    """Clean old log files with security validation"""
    try:
        # Validate and sanitize log directory
        log_path = Path(log_dir).resolve()
        
        # Prevent path traversal - ensure within current working directory
        cwd = Path.cwd()
        try:
            log_path.relative_to(cwd)
        except ValueError:
            logging.warning(f"Path traversal attempt blocked: {log_dir}")
            return
        
        if not log_path.exists() or not log_path.is_dir():
            return
        
        current_time = datetime.now()
        
        for file_path in log_path.glob("*.log"):
            try:
                file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if file_age.days > max_age_days:
                    file_path.unlink()
                    logging.info(f"Deleted old log file: {file_path}")
            except (OSError, ValueError) as e:
                logging.error(f"Error deleting log file {file_path}: {e}")
                
    except Exception as e:
        logging.error(f"Error cleaning old logs: {e}")

def validate_detection_result(result: Dict) -> bool:
    """Validate detection result structure"""
    required_fields = ['authenticity_score', 'confidence', 'classification']
    
    try:
        for field in required_fields:
            if field not in result:
                return False
        
        # Validate score ranges
        if not (0.0 <= result['authenticity_score'] <= 1.0):
            return False
        
        if not (0.0 <= result['confidence'] <= 1.0):
            return False
        
        return True
        
    except Exception:
        # Return False for any validation errors
        return False