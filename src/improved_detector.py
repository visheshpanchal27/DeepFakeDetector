import cv2
import numpy as np
import os
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
import re

warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepfake_detector.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class DetectionResult:
    """Data class for detection results"""
    authenticity_score: float
    confidence: float
    individual_scores: Dict[str, float]
    classification: str
    risk_level: str

class SecurityValidator:
    """Security validation utilities"""
    
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """Validate file path to prevent directory traversal"""
        try:
            resolved_path = Path(file_path).resolve()
            
            # Check file extension
            allowed_extensions = {
                '.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', 
                '.mov', '.webm', '.mkv', '.webp', '.tiff'
            }
            if resolved_path.suffix.lower() not in allowed_extensions:
                return False
            
            # Check if file exists and is actually a file
            if not resolved_path.exists() or not resolved_path.is_file():
                return False
                
            return True
        except Exception as e:
            logging.error(f"Error validating file path {file_path}: {e}")
            return False
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe usage"""
        # Remove dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limit length
        return sanitized[:255]

class ImprovedDeepFakeDetector:
    """Enhanced Deep Fake Detection System with Security and ML"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.face_cascade = None
        self.eye_cascade = None
        self.detection_history: List[Dict] = []
        self._setup_detectors()
        
    def _setup_detectors(self) -> None:
        """Initialize detection models with error handling"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            
            if self.face_cascade.empty() or self.eye_cascade.empty():
                raise ValueError("Failed to load cascade classifiers")
                
            self.logger.info("Detection models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing detectors: {e}")
            raise
    
    def _safe_image_load(self, image_path: str) -> Optional[np.ndarray]:
        """Safely load image with validation"""
        try:
            if not SecurityValidator.validate_file_path(image_path):
                self.logger.warning(f"Invalid file path: {image_path}")
                return None
            
            # Additional path validation
            resolved_path = Path(image_path).resolve()
            
            # Check file size limit (100MB)
            if resolved_path.stat().st_size > 100 * 1024 * 1024:
                self.logger.warning(f"File too large: {image_path}")
                return None
                
            image = cv2.imread(str(resolved_path))
            if image is None:
                self.logger.warning(f"Could not load image: {image_path}")
                return None
                
            # Validate image dimensions
            if image.shape[0] < 32 or image.shape[1] < 32:
                self.logger.warning(f"Image too small: {image.shape}")
                return None
            
            # Validate maximum dimensions
            if image.shape[0] > 4096 or image.shape[1] > 4096:
                self.logger.info(f"Resizing large image: {image.shape}")
                scale = min(4096 / image.shape[0], 4096 / image.shape[1])
                new_h, new_w = int(image.shape[0] * scale), int(image.shape[1] * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def _extract_face_regions(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple]]:
        """Extract face regions with error handling"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            face_regions = []
            for (x, y, w, h) in faces:
                # Add padding with bounds checking
                padding = 20
                y1 = max(0, y - padding)
                y2 = min(image.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(image.shape[1], x + w + padding)
                
                face_roi = image[y1:y2, x1:x2]
                if face_roi.size > 0:
                    face_regions.append(face_roi)
            
            return face_regions, faces
            
        except Exception as e:
            self.logger.error(f"Error extracting faces: {e}")
            return [], []
    
    def _analyze_texture_patterns(self, image: np.ndarray) -> float:
        """Advanced texture analysis for AI detection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale gradient analysis
            gradients = []
            for ksize in [3, 5, 7]:
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
                gradients.append(np.sqrt(grad_x**2 + grad_y**2))
            
            # AI tends to have smoother gradients across scales
            gradient_consistency = np.corrcoef([g.flatten()[:10000] for g in gradients])
            consistency_score = np.mean(gradient_consistency[np.triu_indices_from(gradient_consistency, k=1)])
            
            # Local Binary Pattern analysis
            def lbp_histogram(img, radius=1, n_points=8):
                h, w = img.shape
                lbp = np.zeros_like(img)
                for i in range(radius, h-radius):
                    for j in range(radius, w-radius):
                        center = img[i, j]
                        code = 0
                        for k in range(n_points):
                            angle = 2 * np.pi * k / n_points
                            x = int(i + radius * np.cos(angle))
                            y = int(j + radius * np.sin(angle))
                            if img[x, y] >= center:
                                code |= (1 << k)
                        lbp[i, j] = code
                return np.histogram(lbp, bins=256)[0]
            
            # Sample patches for LBP analysis
            h, w = gray.shape
            patches = [
                gray[i:i+64, j:j+64] 
                for i in range(0, h-64, 128) 
                for j in range(0, w-64, 128)
            ][:9]
            
            if patches:
                lbp_histograms = [lbp_histogram(patch) for patch in patches]
                lbp_variance = np.var([np.var(hist) for hist in lbp_histograms])
                texture_uniformity = 1.0 - min(1.0, lbp_variance / 10000)
            else:
                texture_uniformity = 0.5
            
            # Frequency domain texture analysis
            f_transform = np.fft.fft2(gray)
            magnitude = np.abs(f_transform)
            high_freq_ratio = (
                np.sum(magnitude[magnitude.shape[0]//3:, magnitude.shape[1]//3:]) / 
                np.sum(magnitude)
            )
            
            # AI images often lack high-frequency natural texture
            natural_texture_score = (
                (1.0 - consistency_score) * 0.4 +  # Less consistent = more natural
                texture_uniformity * 0.3 +         # Some uniformity is AI-like
                min(1.0, high_freq_ratio * 3) * 0.3  # Natural images have more high freq
            )
            
            return max(0.0, min(1.0, natural_texture_score))
            
        except Exception as e:
            self.logger.error(f"Error in texture analysis: {e}")
            return 0.5
    
    def _analyze_frequency_domain(self, image: np.ndarray) -> float:
        """Advanced frequency domain analysis for AI artifacts"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Resize for consistent analysis
            if gray.shape[0] > 512 or gray.shape[1] > 512:
                gray = cv2.resize(gray, (512, 512))
            
            # Multi-channel frequency analysis
            channels = cv2.split(image)
            freq_scores = []
            
            for channel in channels:
                if channel.shape[0] > 512 or channel.shape[1] > 512:
                    channel = cv2.resize(channel, (512, 512))
                
                # DCT analysis (JPEG-like compression artifacts)
                dct = cv2.dct(channel.astype(np.float32))
                
                # AI images often have specific DCT patterns
                h, w = dct.shape
                low_freq = dct[:h//4, :w//4]
                mid_freq = dct[h//4:h//2, w//4:w//2]
                high_freq = dct[h//2:, w//2:]
                
                low_energy = np.sum(np.abs(low_freq))
                mid_energy = np.sum(np.abs(mid_freq))
                high_energy = np.sum(np.abs(high_freq))
                total = low_energy + mid_energy + high_energy
                
                if total > 0:
                    # Natural images have more balanced frequency distribution
                    freq_balance = (
                        1.0 - abs(0.6 - (low_energy/total)) - 
                        abs(0.3 - (mid_energy/total))
                    )
                    freq_scores.append(max(0.0, freq_balance))
            
            # FFT analysis for periodic patterns (AI artifacts)
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Look for regular patterns in frequency domain
            h, w = magnitude.shape
            center_h, center_w = h//2, w//2
            
            # Create radial frequency bins
            y, x = np.ogrid[:h, :w]
            distances = np.sqrt((x - center_w)**2 + (y - center_h)**2)
            
            # Analyze frequency distribution
            radial_profile = []
            for r in range(0, min(h, w)//4, 10):
                mask = (distances >= r) & (distances < r + 10)
                if np.any(mask):
                    radial_profile.append(np.mean(magnitude[mask]))
            
            # AI images often have unnatural frequency distributions
            if len(radial_profile) > 3:
                freq_variance = np.var(radial_profile)
                natural_freq_score = min(1.0, freq_variance / 1000000)
            else:
                natural_freq_score = 0.5
            
            # Combine all frequency metrics
            if freq_scores:
                avg_channel_score = np.mean(freq_scores)
                final_score = (avg_channel_score * 0.7 + natural_freq_score * 0.3)
            else:
                final_score = natural_freq_score
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Error in frequency analysis: {e}")
            return 0.5
    
    def _detect_facial_inconsistencies(self, image: np.ndarray) -> float:
        """Advanced facial geometry and consistency analysis"""
        try:
            face_regions, faces = self._extract_face_regions(image)
            
            if not face_regions:
                return 0.5
            
            consistency_scores = []
            
            for face_roi in face_regions:
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                h, w = gray_face.shape
                
                if h < 64 or w < 64:  # Skip too small faces
                    continue
                
                # Advanced symmetry analysis with multiple methods
                mid = w // 2
                left_half = gray_face[:, :mid]
                right_half = cv2.flip(gray_face[:, mid:], 1)
                
                # Ensure same dimensions
                min_width = min(left_half.shape[1], right_half.shape[1])
                min_height = min(left_half.shape[0], right_half.shape[0])
                left_half = left_half[:min_height, :min_width]
                right_half = right_half[:min_height, :min_width]
                
                # Multiple symmetry metrics
                pixel_diff = np.mean(cv2.absdiff(left_half, right_half)) / 255.0
                correlation = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0,0]
                
                symmetry_score = (1.0 - pixel_diff) * 0.6 + correlation * 0.4
                
                # Eye detection and analysis
                eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 5, minSize=(10, 10))
                
                eye_score = 0.5
                if len(eyes) >= 2:
                    # Analyze eye positioning and symmetry
                    eye_centers = [(x + w//2, y + h//2) for x, y, w, h in eyes[:2]]
                    
                    if len(eye_centers) == 2:
                        # Check if eyes are at similar height (horizontal alignment)
                        height_diff = abs(eye_centers[0][1] - eye_centers[1][1])
                        max_height_diff = h * 0.1  # Allow 10% variation
                        
                        # Check eye spacing consistency
                        eye_distance = abs(eye_centers[0][0] - eye_centers[1][0])
                        expected_distance = w * 0.3  # Typical eye distance
                        if max(eye_distance, expected_distance) > 0:
                            distance_ratio = min(eye_distance, expected_distance) / max(eye_distance, expected_distance)
                        else:
                            distance_ratio = 0.5
                        
                        if max_height_diff > 0:
                            alignment_score = 1.0 - min(1.0, height_diff / max_height_diff)
                        else:
                            alignment_score = 1.0
                        spacing_score = distance_ratio
                        
                        eye_score = (alignment_score * 0.6 + spacing_score * 0.4)
                elif len(eyes) == 1:
                    eye_score = 0.3  # Single eye detected
                else:
                    eye_score = 0.1  # No eyes detected
                
                # Facial feature gradient analysis
                # AI faces often have unnatural gradient transitions
                sobel_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                
                # Natural faces have varied gradient patterns
                gradient_variance = np.var(gradient_magnitude)
                gradient_score = min(1.0, gradient_variance / 1000)
                
                # Skin texture analysis
                # AI skin often lacks natural pore/texture patterns
                blur_diff = cv2.absdiff(gray_face, cv2.GaussianBlur(gray_face, (5, 5), 0))
                texture_detail = np.mean(blur_diff)
                texture_score = min(1.0, texture_detail / 10)
                
                # Combine all facial metrics
                face_consistency = (
                    symmetry_score * 0.35 +
                    eye_score * 0.25 +
                    gradient_score * 0.25 +
                    texture_score * 0.15
                )
                
                consistency_scores.append(max(0.0, min(1.0, face_consistency)))
            
            return np.mean(consistency_scores) if consistency_scores else 0.5
            
        except Exception as e:
            self.logger.error(f"Error in facial analysis: {e}")
            return 0.5
    
    def _detect_ai_signatures(self, image: np.ndarray, file_path: str = "") -> float:
        """Advanced AI signature detection with ML artifact patterns"""
        try:
            filename = os.path.basename(file_path).lower()
            
            # Enhanced filename analysis with AI model signatures
            ai_keywords = {
                'dalle': 0.9, 'midjourney': 0.9, 'stable': 0.8, 'gemini': 0.8,
                'firefly': 0.8, 'leonardo': 0.7, 'runway': 0.7, 'synthesia': 0.8,
                'generated': 0.6, 'artificial': 0.6, 'synthetic': 0.6, 'ai': 0.5,
                'deepfake': 0.9, 'faceswap': 0.9, 'gpt': 0.4, 'claude': 0.4
            }
            
            filename_score = 0.0
            for keyword, weight in ai_keywords.items():
                if keyword in filename:
                    filename_score = max(filename_score, weight)
            
            # Advanced watermark and signature detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Multi-location watermark check
            watermark_score = 0.0
            locations = [
                (slice(-h//8, None), slice(-w//8, None)),  # Bottom-right
                (slice(None, h//8), slice(-w//8, None)),   # Top-right
                (slice(-h//8, None), slice(None, w//8)),   # Bottom-left
                (slice(None, h//8), slice(None, w//8))     # Top-left
            ]
            
            for loc in locations:
                corner = gray[loc]
                if corner.size > 100:  # Ensure minimum size
                    corner_var = np.var(corner)
                    global_var = np.var(gray)
                    
                    # Check for suspiciously smooth areas (watermarks)
                    if corner_var < global_var * 0.03:
                        watermark_score += 0.25
                    
                    # Check for text-like patterns (AI signatures)
                    edges = cv2.Canny(corner, 50, 150)
                    edge_density = np.sum(edges > 0) / edges.size
                    if 0.01 < edge_density < 0.05:  # Text-like edge density
                        watermark_score += 0.2
            
            # AI artifact pattern detection
            artifact_score = 0.0
            
            # 1. Checkerboard artifacts (common in GANs)
            kernel_checker = np.array([[1, -1], [-1, 1]], dtype=np.float32)
            checker_response = cv2.filter2D(gray.astype(np.float32), -1, kernel_checker)
            checker_energy = np.mean(np.abs(checker_response))
            if checker_energy > 15:  # Strong checkerboard pattern
                artifact_score += 0.3
            
            # 2. Upsampling artifacts (bilinear/bicubic patterns)
            # Downsample and upsample to detect interpolation artifacts
            if h > 128 and w > 128:
                small = cv2.resize(gray, (w//4, h//4), interpolation=cv2.INTER_AREA)
                upsampled = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
                upsample_diff = cv2.absdiff(gray, upsampled)
                upsample_similarity = 1.0 - (np.mean(upsample_diff) / 255.0)
                
                if upsample_similarity > 0.85:  # Too similar to simple upsampling
                    artifact_score += 0.4
            
            # 3. Spectral analysis for AI generation patterns
            # AI models often produce specific frequency signatures
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log(np.abs(f_shift) + 1)
            
            # Look for regular patterns in frequency domain
            h_freq, w_freq = magnitude.shape
            center_h, center_w = h_freq//2, w_freq//2
            
            # Check for artificial regularity in frequency domain
            radial_samples = []
            for angle in np.linspace(0, 2*np.pi, 16):
                for radius in range(10, min(center_h, center_w)//2, 5):
                    y = int(center_h + radius * np.sin(angle))
                    x = int(center_w + radius * np.cos(angle))
                    if 0 <= y < h_freq and 0 <= x < w_freq:
                        radial_samples.append(magnitude[y, x])
            
            if radial_samples:
                mean_val = np.mean(radial_samples)
                if mean_val > 0:
                    freq_regularity = 1.0 - (np.var(radial_samples) / mean_val)
                    if freq_regularity > 0.7:  # Too regular frequency pattern
                        artifact_score += 0.3
            
            # 4. Color distribution analysis
            # AI images often have unnatural color distributions
            if len(image.shape) == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
                sat_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
                
                # Check for unnatural hue clustering
                hue_peaks = np.where(hue_hist > np.max(hue_hist) * 0.1)[0]
                if len(hue_peaks) < 5:  # Too few dominant hues
                    artifact_score += 0.2
                
                # Check saturation distribution
                sat_mean = np.mean(sat_hist)
                sat_std = np.std(sat_hist)
                if sat_mean > 0 and sat_std < sat_mean * 0.3:  # Too uniform saturation
                    artifact_score += 0.2
            
            # Combine all AI detection scores
            total_ai_score = min(1.0, filename_score + watermark_score + artifact_score)
            
            # Return human probability (inverse of AI score)
            return max(0.0, 1.0 - total_ai_score)
            
        except Exception as e:
            self.logger.error(f"Error in AI signature detection: {e}")
            return 0.5
    
    def _analyze_compression_artifacts(self, image: np.ndarray) -> float:
        """Analyze compression patterns"""
        try:
            # Convert to YUV for JPEG analysis
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y_channel = yuv[:, :, 0].astype(np.float32)
            
            # DCT block analysis
            h, w = y_channel.shape
            block_size = 8
            compression_scores = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = y_channel[i:i+block_size, j:j+block_size]
                    
                    # Apply DCT
                    dct_block = cv2.dct(block)
                    
                    # Analyze high frequency components
                    high_freq = dct_block[4:, 4:]
                    high_freq_energy = np.sum(np.abs(high_freq))
                    total_energy = np.sum(np.abs(dct_block))
                    
                    if total_energy > 0:
                        compression_scores.append(high_freq_energy / total_energy)
            
            if not compression_scores:
                return 0.5
            
            avg_compression = np.mean(compression_scores)
            compression_variance = np.var(compression_scores)
            
            # Natural images have moderate compression with some variance
            naturalness = 1.0 - abs(0.3 - avg_compression) - abs(0.05 - compression_variance)
            
            return max(0.0, min(1.0, naturalness))
            
        except Exception as e:
            self.logger.error(f"Error in compression analysis: {e}")
            return 0.5
    
    def analyze_image(self, image_path: str) -> Optional[DetectionResult]:
        """Analyze single image for AI vs Human classification"""
        try:
            self.logger.info(f"Analyzing image: {image_path}")
            
            # Load and validate image
            print("Loading image...")
            image = self._safe_image_load(image_path)
            if image is None:
                return None
            
            # Perform individual analyses with progress
            print("Analyzing texture...")
            texture_score = self._analyze_texture_patterns(image)
            
            print("Analyzing frequency domain...")
            frequency_score = self._analyze_frequency_domain(image)
            
            print("Analyzing facial features...")
            facial_score = self._detect_facial_inconsistencies(image)
            
            print("Detecting AI signatures...")
            ai_signature_score = self._detect_ai_signatures(image, image_path)
            
            print("Analyzing compression...")
            compression_score = self._analyze_compression_artifacts(image)
            
            # Store individual scores
            individual_scores = {
                'texture_analysis': texture_score,
                'frequency_domain': frequency_score,
                'facial_consistency': facial_score,
                'ai_signatures': ai_signature_score,
                'compression_artifacts': compression_score
            }
            
            # Optimized weights for advanced algorithms
            weights = {
                'texture_analysis': 0.30,      # Enhanced multi-scale analysis
                'frequency_domain': 0.25,      # Advanced DCT + FFT analysis
                'facial_consistency': 0.20,    # Improved geometric analysis
                'ai_signatures': 0.20,         # ML artifact detection
                'compression_artifacts': 0.05   # Less critical with new methods
            }
            
            authenticity_score = sum(
                individual_scores[metric] * weights[metric] 
                for metric in weights.keys()
            )
            
            # Calculate confidence
            confidence = min(0.95, 0.5 + abs(authenticity_score - 0.5))
            
            # Enhanced classification with stricter thresholds
            if authenticity_score > 0.85:
                classification = "AUTHENTIC_HUMAN"
                risk_level = "SAFE"
            elif authenticity_score > 0.70:
                classification = "HUMAN_ENHANCED"
                risk_level = "LOW"
            elif authenticity_score > 0.30:
                classification = "SUSPICIOUS"
                risk_level = "MEDIUM"
            else:
                classification = "AI_GENERATED"
                risk_level = "HIGH"
            
            result = DetectionResult(
                authenticity_score=authenticity_score,
                confidence=confidence,
                individual_scores=individual_scores,
                classification=classification,
                risk_level=risk_level
            )
            
            # Store in history
            self.detection_history.append({
                'file': image_path,
                'score': authenticity_score,
                'classification': classification,
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.info(f"Analysis complete: {classification} ({authenticity_score:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing image {image_path}: {e}")
            return None
    
    def analyze_video(self, video_path: str) -> Optional[DetectionResult]:
        """Analyze video for deepfake detection"""
        try:
            self.logger.info(f"Analyzing video: {video_path}")
            
            # Validate video path
            if not SecurityValidator.validate_file_path(video_path):
                self.logger.warning(f"Invalid video path: {video_path}")
                return None
            
            resolved_path = Path(video_path).resolve()
            
            # Check file size limit (500MB for videos)
            if resolved_path.stat().st_size > 500 * 1024 * 1024:
                self.logger.warning(f"Video file too large: {video_path}")
                return None
            
            print("Loading video...")
            cap = cv2.VideoCapture(str(resolved_path))
            
            if not cap.isOpened():
                self.logger.warning(f"Could not open video: {video_path}")
                return None
            
            frame_scores = []
            frame_count = 0
            max_frames = 30  # Analyze first 30 frames
            
            print("Analyzing video frames...")
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analyze frame as image
                texture_score = self._analyze_texture_patterns(frame)
                frequency_score = self._analyze_frequency_domain(frame)
                facial_score = self._detect_facial_inconsistencies(frame)
                compression_score = self._analyze_compression_artifacts(frame)
                
                # Video-specific AI detection (no filename analysis for frames)
                ai_signature_score = 0.5  # Neutral for individual frames
                
                individual_scores = {
                    'texture_analysis': texture_score,
                    'frequency_domain': frequency_score,
                    'facial_consistency': facial_score,
                    'ai_signatures': ai_signature_score,
                    'compression_artifacts': compression_score
                }
                
                # Calculate frame authenticity
                weights = {
                    'texture_analysis': 0.30,
                    'frequency_domain': 0.25,
                    'facial_consistency': 0.20,
                    'ai_signatures': 0.15,  # Lower weight for video frames
                    'compression_artifacts': 0.10
                }
                
                frame_authenticity = sum(
                    individual_scores[metric] * weights[metric] 
                    for metric in weights.keys()
                )
                
                frame_scores.append(frame_authenticity)
                frame_count += 1
            
            cap.release()
            
            if not frame_scores:
                return None
            
            # Analyze filename for video-level AI detection
            filename = os.path.basename(video_path).lower()
            video_ai_score = 0.5
            
            ai_keywords_strong = ['ai', 'generated', 'deepfake', 'synthetic']
            if any(keyword in filename for keyword in ai_keywords_strong):
                video_ai_score = 0.2  # Strong AI indicator
            elif any(keyword in filename for keyword in ['fake', 'artificial']):
                video_ai_score = 0.3  # Moderate AI indicator
            
            # Combine frame analysis with video-level analysis
            avg_frame_score = np.mean(frame_scores)
            frame_consistency = 1.0 - np.std(frame_scores)  # More consistent = more suspicious
            
            # Final video authenticity score
            authenticity_score = (
                avg_frame_score * 0.6 +
                video_ai_score * 0.3 +
                frame_consistency * 0.1
            )
            
            # Calculate confidence
            confidence = min(0.95, 0.5 + abs(authenticity_score - 0.5))
            
            # Enhanced classification with stricter thresholds
            if authenticity_score > 0.85:
                classification = "AUTHENTIC_HUMAN"
                risk_level = "SAFE"
            elif authenticity_score > 0.70:
                classification = "HUMAN_ENHANCED"
                risk_level = "LOW"
            elif authenticity_score > 0.30:
                classification = "SUSPICIOUS"
                risk_level = "MEDIUM"
            else:
                classification = "AI_GENERATED"
                risk_level = "HIGH"
            
            # Store individual scores for video
            individual_scores = {
                'frame_analysis': avg_frame_score,
                'filename_analysis': video_ai_score,
                'temporal_consistency': frame_consistency,
                'overall_assessment': authenticity_score
            }
            
            result = DetectionResult(
                authenticity_score=authenticity_score,
                confidence=confidence,
                individual_scores=individual_scores,
                classification=classification,
                risk_level=risk_level
            )
            
            # Store in history
            self.detection_history.append({
                'file': video_path,
                'score': authenticity_score,
                'classification': classification,
                'timestamp': datetime.now().isoformat(),
                'type': 'video'
            })
            
            self.logger.info(f"Video analysis complete: {classification} ({authenticity_score:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing video {video_path}: {e}")
            return None
    
    def batch_analyze(self, directory_path: str) -> List[DetectionResult]:
        """Analyze multiple files in directory with security validation"""
        try:
            # Validate directory path
            resolved_dir = Path(directory_path).resolve()
            
            # Allow directories from anywhere
            if '..' in directory_path:
                self.logger.error(f"Suspicious path blocked: {directory_path}")
                return []
            
            if not resolved_dir.exists() or not resolved_dir.is_dir():
                self.logger.error(f"Directory not found: {directory_path}")
                return []
            
            supported_formats = {
                '.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi',
                '.mov', '.webm', '.mkv', '.webp', '.tiff'
            }
            results = []
            processed_count = 0
            max_files = 1000  # Limit batch size
            
            for file_path in resolved_dir.rglob('*'):
                if processed_count >= max_files:
                    self.logger.warning(f"Batch size limit reached: {max_files}")
                    break
                
                if file_path.suffix.lower() in supported_formats and file_path.is_file():
                    # Additional security validation
                    if not SecurityValidator.validate_file_path(str(file_path)):
                        continue
                    
                    # Check if it's a video or image
                    video_formats = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
                    if file_path.suffix.lower() in video_formats:
                        result = self.analyze_video(str(file_path))
                    else:
                        result = self.analyze_image(str(file_path))
                    
                    if result:
                        results.append(result)
                    
                    processed_count += 1
            
            self.logger.info(f"Batch analysis complete: {len(results)} files processed")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch analysis: {e}")
            return []
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive analysis report with security validation"""
        try:
            if not self.detection_history:
                self.logger.warning("No analysis history available for report")
                return ""
            
            # Use organized reports directory with validation
            reports_dir = Path("AI_Detection_Reports")
            reports_dir.mkdir(exist_ok=True)
            
            # Validate output path if provided
            if output_path:
                output_path = Path(output_path).resolve()
                # Ensure output is within current directory
                cwd = Path.cwd()
                try:
                    output_path.relative_to(cwd)
                except ValueError:
                    self.logger.warning(f"Invalid output path: {output_path}")
                    output_path = None
            
            # Generate report data
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            total_files = len(self.detection_history)
            
            classifications = [h['classification'] for h in self.detection_history]
            scores = [h['score'] for h in self.detection_history]
            
            # Statistics
            stats = {
                'total_files': total_files,
                'avg_score': np.mean(scores),
                'authentic_count': classifications.count('AUTHENTIC_HUMAN'),
                'enhanced_count': classifications.count('HUMAN_ENHANCED'),
                'suspicious_count': classifications.count('SUSPICIOUS'),
                'ai_count': classifications.count('AI_GENERATED')
            }
            
            # Generate HTML report
            if output_path is None:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Sanitize filename
                safe_filename = f"AI_Detection_Report_{timestamp_str}.html"
                output_path = reports_dir / safe_filename
            
            html_content = self._generate_html_report(stats, timestamp)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Generate JSON data
            json_path = output_path.with_suffix('.json')
            json_data = {
                'timestamp': timestamp,
                'statistics': stats,
                'files': [{
                    'file': os.path.basename(h['file']),  # Only basename for security
                    'score': h['score'],
                    'classification': h['classification'],
                    'timestamp': h['timestamp']
                } for h in self.detection_history]
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            
            self.logger.info(f"Report generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return ""
    
    def _generate_html_report(self, stats: Dict, timestamp: str) -> str:
        """Generate HTML report content"""
        # Build file results HTML
        file_results_html = []
        for i, entry in enumerate(self.detection_history, 1):
            score = entry['score']
            classification = entry['classification']
            
            # Determine styling based on classification
            if classification == 'AUTHENTIC_HUMAN':
                css_class = 'authentic'
                status_text = '✅ AUTHENTIC HUMAN'
                description = 'Natural human-created content'
            elif classification == 'HUMAN_ENHANCED':
                css_class = 'enhanced'
                status_text = '🔄 HUMAN ENHANCED'
                description = 'Human content with digital enhancement'
            elif classification == 'SUSPICIOUS':
                css_class = 'suspicious'
                status_text = '⚠️ SUSPICIOUS'
                description = 'Potentially AI-generated or heavily processed'
            else:
                css_class = 'ai-generated'
                status_text = '🤖 AI GENERATED'
                description = 'Artificial intelligence generated content'
            
            file_html = f"""
            <div class="file-result {css_class}">
                <div class="file-header">
                    <h3>#{i}: {os.path.basename(entry['file'])}</h3>
                    <span class="status-badge {css_class}">{status_text}</span>
                </div>
                <div class="score-display">
                    <div class="score-circle {css_class}">
                        <span class="score-value">{score:.0%}</span>
                        <span class="score-label">Human</span>
                    </div>
                    <div class="score-details">
                        <p><strong>Classification:</strong> {classification.replace('_', ' ')}</p>
                        <p><strong>Description:</strong> {description}</p>
                        <p><strong>File:</strong> {entry['file']}</p>
                    </div>
                </div>
            </div>
            """
            file_results_html.append(file_html)
        
        # Complete HTML template
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Detection Analysis Report</title>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 40px 20px; }}
                .container {{ max-width: 1400px; margin: 0 auto; }}
                
                .header {{ background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 50px 40px; border-radius: 20px; margin-bottom: 40px; text-align: center; box-shadow: 0 20px 40px rgba(0,0,0,0.1); }}
                .header h1 {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3.5em; font-weight: 700; margin-bottom: 15px; }}
                .header .subtitle {{ color: #6c757d; font-size: 1.3em; font-weight: 400; margin-bottom: 10px; }}
                .header .timestamp {{ color: #9ca3af; font-size: 1.1em; font-weight: 300; }}
                
                .stats-overview {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 30px; margin-bottom: 50px; }}
                .stat-card {{ background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 35px 30px; border-radius: 20px; text-align: center; box-shadow: 0 15px 35px rgba(0,0,0,0.08); transition: transform 0.3s ease; }}
                .stat-card:hover {{ transform: translateY(-5px); box-shadow: 0 25px 50px rgba(0,0,0,0.15); }}
                .stat-number {{ font-size: 3.2em; font-weight: 700; margin-bottom: 12px; }}
                .stat-label {{ color: #6c757d; font-size: 1.2em; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }}
                .authentic .stat-number {{ color: #27ae60; }}
                .enhanced .stat-number {{ color: #3498db; }}
                .suspicious .stat-number {{ color: #f39c12; }}
                .ai-generated .stat-number {{ color: #e74c3c; }}
                
                .file-result {{ background: white; margin: 20px 0; border-radius: 15px; 
                               overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
                .file-result.authentic {{ border-left: 5px solid #27ae60; }}
                .file-result.enhanced {{ border-left: 5px solid #3498db; }}
                .file-result.suspicious {{ border-left: 5px solid #f39c12; }}
                .file-result.ai-generated {{ border-left: 5px solid #e74c3c; }}
                
                .file-header {{ display: flex; justify-content: space-between; align-items: center; 
                               padding: 20px; background: #f8f9fa; }}
                .file-header h3 {{ color: #2c3e50; }}
                .status-badge {{ padding: 8px 16px; border-radius: 20px; color: white; font-weight: bold; }}
                .status-badge.authentic {{ background: #27ae60; }}
                .status-badge.enhanced {{ background: #3498db; }}
                .status-badge.suspicious {{ background: #f39c12; }}
                .status-badge.ai-generated {{ background: #e74c3c; }}
                
                .score-display {{ display: flex; align-items: center; gap: 30px; padding: 20px; }}
                .score-circle {{ width: 80px; height: 80px; border-radius: 50%; 
                                display: flex; flex-direction: column; align-items: center; 
                                justify-content: center; color: white; }}
                .score-circle.authentic {{ background: #27ae60; }}
                .score-circle.enhanced {{ background: #3498db; }}
                .score-circle.suspicious {{ background: #f39c12; }}
                .score-circle.ai-generated {{ background: #e74c3c; }}
                .score-value {{ font-size: 1.5em; font-weight: bold; }}
                .score-label {{ font-size: 0.8em; }}
                .score-details {{ flex: 1; }}
                .score-details p {{ margin: 5px 0; color: #2c3e50; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>AI Detection Analysis</h1>
                    <p class="subtitle">Professional Content Authenticity Report</p>
                    <p class="timestamp">Generated on {timestamp}</p>
                </div>
                
                <div class="stats-overview">
                    <div class="stat-card">
                        <div class="stat-number">{stats['total_files']}</div>
                        <div class="stat-label">Total Files</div>
                    </div>
                    <div class="stat-card authentic">
                        <div class="stat-number">{stats['authentic_count']}</div>
                        <div class="stat-label">Authentic Human</div>
                    </div>
                    <div class="stat-card enhanced">
                        <div class="stat-number">{stats['enhanced_count']}</div>
                        <div class="stat-label">Human Enhanced</div>
                    </div>
                    <div class="stat-card suspicious">
                        <div class="stat-number">{stats['suspicious_count']}</div>
                        <div class="stat-label">Suspicious</div>
                    </div>
                    <div class="stat-card ai-generated">
                        <div class="stat-number">{stats['ai_count']}</div>
                        <div class="stat-label">AI Generated</div>
                    </div>
                </div>
                
                <div class="results-section">
                    <h2 style="color: white; margin-bottom: 20px; text-align: center;">📋 Detailed Analysis Results</h2>
                    {''.join(file_results_html)}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template

def main():
    """Main function with improved error handling"""
    try:
        detector = ImprovedDeepFakeDetector()
        
        print("🔍 ENHANCED AI DETECTION SYSTEM")
        print("=" * 50)
        print("🛡️  Security Enhanced | 🧠 ML Powered | 📊 Professional Reports")
        print("=" * 50)
        
        while True:
            print("\n📋 MENU OPTIONS:")
            print("1. 🖼️  Analyze Single Image")
            print("2. 🎥 Analyze Video File")
            print("3. 📁 Batch Analyze Directory")
            print("4. 📄 Generate Report")
            print("5. 📊 View Statistics")
            print("6. 🚪 Exit")
            
            try:
                choice = input("\nEnter choice (1-6): ").strip()
                
                if choice == '1':
                    path = input("Enter image path: ").strip()
                    result = detector.analyze_image(path)
                    if result:
                        print(f"\n✅ Analysis Complete!")
                        print(f"Classification: {result.classification}")
                        print(f"Authenticity Score: {result.authenticity_score:.1%}")
                        print(f"Confidence: {result.confidence:.1%}")
                    else:
                        print("❌ Analysis failed. Check file path and format.")
                
                elif choice == '2':
                    path = input("Enter video path: ").strip()
                    result = detector.analyze_video(path)
                    if result:
                        print(f"\n✅ Video Analysis Complete!")
                        print(f"Classification: {result.classification}")
                        print(f"Authenticity Score: {result.authenticity_score:.1%}")
                        print(f"Confidence: {result.confidence:.1%}")
                    else:
                        print("❌ Video analysis failed. Check file path and format.")
                
                elif choice == '3':
                    path = input("Enter directory path: ").strip()
                    results = detector.batch_analyze(path)
                    if results:
                        print(f"\n✅ Batch analysis complete! Processed {len(results)} files.")
                        # Show summary
                        classifications = [r.classification for r in results]
                        for cls in set(classifications):
                            count = classifications.count(cls)
                            print(f"  {cls.replace('_', ' ')}: {count}")
                    else:
                        print("❌ No files processed. Check directory path.")
                
                elif choice == '4':
                    report_path = detector.generate_report()
                    if report_path:
                        print(f"✅ Report generated: {report_path}")
                    else:
                        print("❌ No analysis history available.")
                
                elif choice == '5':
                    if detector.detection_history:
                        total = len(detector.detection_history)
                        scores = [h['score'] for h in detector.detection_history]
                        avg_score = np.mean(scores)
                        print(f"\n📊 STATISTICS:")
                        print(f"Total Files Analyzed: {total}")
                        print(f"Average Authenticity Score: {avg_score:.1%}")
                        print(f"Latest Analysis: {detector.detection_history[-1]['file']}")
                    else:
                        print("📊 No analysis history available.")
                
                elif choice == '6':
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid choice. Please enter 1-6.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()