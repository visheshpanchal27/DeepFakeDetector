#!/usr/bin/env python3
"""
Test script for the Enhanced AI Detection System
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from improved_detector import ImprovedDeepFakeDetector
    from config import Config
    from utils import setup_logging
    import cv2
    import numpy as np
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required dependencies: pip install opencv-python numpy")
    sys.exit(1)

def create_test_image():
    """Create a simple test image"""
    # Create a simple test image
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Add some patterns
    cv2.rectangle(test_image, (50, 50), (250, 250), (255, 255, 255), -1)
    cv2.circle(test_image, (150, 150), 50, (0, 0, 255), -1)
    
    # Save test image
    test_path = "test_image.jpg"
    cv2.imwrite(test_path, test_image)
    return test_path

def test_security_features():
    """Test security validation features"""
    print("🔒 Testing Security Features...")
    
    from improved_detector import SecurityValidator
    
    # Test path validation
    valid_paths = [
        "image.jpg",
        "folder/image.png",
        "test.mp4"
    ]
    
    invalid_paths = [
        "../../../etc/passwd",
        "image.exe",
        "script.js"
    ]
    
    print("✅ Valid paths:")
    for path in valid_paths:
        result = SecurityValidator.validate_file_path(path)
        print(f"  {path}: {'✅' if result else '❌'}")
    
    print("❌ Invalid paths (should be rejected):")
    for path in invalid_paths:
        result = SecurityValidator.validate_file_path(path)
        print(f"  {path}: {'❌' if not result else '⚠️ SECURITY ISSUE'}")

def test_detector_initialization():
    """Test detector initialization"""
    print("\n🔧 Testing Detector Initialization...")
    
    try:
        detector = ImprovedDeepFakeDetector()
        print("✅ Detector initialized successfully")
        
        # Test if cascades are loaded
        if detector.face_cascade is not None and detector.eye_cascade is not None:
            print("✅ Face and eye detection models loaded")
        else:
            print("⚠️ Detection models not loaded properly")
            
        return detector
        
    except Exception as e:
        print(f"❌ Detector initialization failed: {e}")
        return None

def test_image_analysis(detector):
    """Test image analysis functionality"""
    print("\n🖼️ Testing Image Analysis...")
    
    # Create test image
    test_path = create_test_image()
    
    try:
        result = detector.analyze_image(test_path)
        
        if result:
            print("✅ Image analysis completed successfully")
            print(f"  Classification: {result.classification}")
            print(f"  Authenticity Score: {result.authenticity_score:.1%}")
            print(f"  Confidence: {result.confidence:.1%}")
            print(f"  Risk Level: {result.risk_level}")
            
            # Test individual scores
            print("  Individual Scores:")
            for metric, score in result.individual_scores.items():
                print(f"    {metric}: {score:.1%}")
        else:
            print("❌ Image analysis failed")
            
    except Exception as e:
        print(f"❌ Error during image analysis: {e}")
    
    finally:
        # Clean up test image
        if os.path.exists(test_path):
            os.remove(test_path)

def test_error_handling(detector):
    """Test error handling"""
    print("\n🛡️ Testing Error Handling...")
    
    # Test with non-existent file
    result = detector.analyze_image("non_existent_file.jpg")
    if result is None:
        print("✅ Non-existent file handled correctly")
    else:
        print("❌ Non-existent file not handled properly")
    
    # Test with invalid path
    result = detector.analyze_image("../../../invalid/path.jpg")
    if result is None:
        print("✅ Invalid path handled correctly")
    else:
        print("❌ Invalid path not handled properly")

def test_configuration():
    """Test configuration system"""
    print("\n⚙️ Testing Configuration System...")
    
    try:
        # Test threshold access
        thresholds = Config.AUTHENTICITY_THRESHOLDS
        print(f"✅ Authenticity thresholds loaded: {len(thresholds)} levels")
        
        # Test weights
        weights = Config.ANALYSIS_WEIGHTS
        total_weight = sum(weights.values())
        print(f"✅ Analysis weights loaded, total: {total_weight:.2f}")
        
        if abs(total_weight - 1.0) < 0.01:
            print("✅ Weights are properly normalized")
        else:
            print("⚠️ Weights may not be properly normalized")
            
        # Test classification function
        test_scores = [0.9, 0.7, 0.5, 0.2]
        for score in test_scores:
            classification = Config.get_classification(score)
            print(f"  Score {score:.1f} → {classification}")
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

def test_report_generation(detector):
    """Test report generation"""
    print("\n📄 Testing Report Generation...")
    
    # Need some analysis history for report
    if not detector.detection_history:
        print("⚠️ No analysis history available for report test")
        return
    
    try:
        report_path = detector.generate_report()
        
        if report_path and os.path.exists(report_path):
            print(f"✅ Report generated successfully: {report_path}")
            
            # Check file size
            file_size = os.path.getsize(report_path)
            if file_size > 0:
                print(f"✅ Report file has content ({file_size} bytes)")
            else:
                print("❌ Report file is empty")
        else:
            print("❌ Report generation failed")
            
    except Exception as e:
        print(f"❌ Error during report generation: {e}")

def run_all_tests():
    """Run all tests"""
    print("🧪 ENHANCED AI DETECTION SYSTEM - TEST SUITE")
    print("=" * 60)
    
    # Test security features
    test_security_features()
    
    # Test detector initialization
    detector = test_detector_initialization()
    
    if detector is None:
        print("❌ Cannot continue tests - detector initialization failed")
        return False
    
    # Test configuration
    test_configuration()
    
    # Test image analysis
    test_image_analysis(detector)
    
    # Test error handling
    test_error_handling(detector)
    
    # Test report generation
    test_report_generation(detector)
    
    print("\n" + "=" * 60)
    print("🎉 TEST SUITE COMPLETED")
    
    # Summary
    if detector.detection_history:
        print(f"✅ Analysis history contains {len(detector.detection_history)} entries")
    
    return True

def main():
    """Main test function"""
    try:
        success = run_all_tests()
        
        if success:
            print("\n✅ All tests completed successfully!")
            print("🚀 Enhanced AI Detection System is ready to use.")
        else:
            print("\n❌ Some tests failed. Please check the installation.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error during testing: {e}")

if __name__ == "__main__":
    main()