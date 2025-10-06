#!/usr/bin/env python3
"""
DeepFake Detection System
Author: Vishesh Panchal
GitHub: https://github.com/visheshpanchal27/DeepFakeDetector
Optimized AI vs Human Detection System
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from detector import DeepFakeDetector
    from config import Config
    from utils import setup_logging
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required dependencies: pip install -r requirements.txt")
    sys.exit(1)

def display_banner():
    """Display application banner"""
    print("\n" + "="*50)
    print("🤖 AI vs HUMAN DETECTION SYSTEM")
    print("="*50)
    print("🛡️ Enhanced Security | 🧠 ML Powered | 📊 Reports")
    print("="*50)

def display_menu():
    """Display main menu"""
    print("\n📋 MENU:")
    print("1. 🖼️  Analyze Image")
    print("2. 🎥 Analyze Video") 
    print("3. 📁 Batch Directory")
    print("4. 📄 Generate Report")
    print("5. 📊 Statistics")
    print("6. ⚙️  System Info")
    print("7. 📚 Help Guide")
    print("8. 🚪 Exit")

def display_analysis_result(result):
    """Display comprehensive analysis result"""
    score = result.authenticity_score
    
    # Enhanced result display with color coding
    print("\n" + "="*50)
    if score > 0.75:
        print("✅ MAIN RESULT: CREATED BY HUMAN")
        print(f"📊 Human Confidence: {score:.1%}")
        verdict = "This content appears to be authentic human creation"
    elif score > 0.60:
        print("🔄 MAIN RESULT: CREATED BY HUMAN (Enhanced)")
        print(f"📊 Human Confidence: {score:.1%}")
        verdict = "Human content with possible beauty filters or editing"
    elif score > 0.40:
        print("⚠️ MAIN RESULT: LIKELY CREATED BY AI")
        print(f"📊 AI Probability: {(1-score)*100:.1f}%")
        verdict = "Content shows AI indicators - verify before use"
    else:
        print("🤖 MAIN RESULT: CREATED BY AI")
        print(f"📊 AI Confidence: {(1-score)*100:.1f}%")
        verdict = "This content is likely artificially generated"
    
    print(f"🎯 Classification: {result.classification.replace('_', ' ')}")
    print(f"⚠️  Risk Level: {result.risk_level}")
    print(f"🔍 Verdict: {verdict}")
    print("="*50)
    
    # Enhanced details with explanations
    if input("\n📋 See detailed technical analysis? (y/n): ").strip().lower() in ['y', 'yes']:
        print("\n🔬 TECHNICAL ANALYSIS BREAKDOWN:")
        print("-" * 50)
        
        for metric, score_val in result.individual_scores.items():
            status = "✅ PASS" if score_val > 0.6 else "⚠️ WARN" if score_val > 0.4 else "❌ FAIL"
            metric_name = metric.replace('_', ' ').title()
            
            # Add explanations for each metric
            explanation = get_metric_explanation(metric, score_val)
            print(f"  {status} {metric_name}: {score_val:.1%}")
            print(f"      → {explanation}")
            
        # Overall assessment
        print("\n🎯 OVERALL ASSESSMENT:")
        print("-" * 30)
        if score > 0.75:
            print("✅ HIGH CONFIDENCE: Natural human creation patterns detected")
            print("   Recommendation: Safe to use as authentic content")
        elif score > 0.60:
            print("🔄 MODERATE CONFIDENCE: Human with digital processing")
            print("   Recommendation: Likely safe but verify source if critical")
        elif score > 0.40:
            print("⚠️ LOW CONFIDENCE: Multiple AI indicators present")
            print("   Recommendation: Requires verification before use")
        else:
            print("🚨 VERY LOW CONFIDENCE: Strong AI generation signatures")
            print("   Recommendation: Do not use as authentic human content")

def analyze_single_image(detector):
    """Handle single image analysis with drag-drop support"""
    while True:
        path = input("\n📁 Enter image path (drag & drop supported) or 'back': ").strip().strip('"')
        
        if path.lower() == 'back':
            break
            
        if not path:
            print("❌ Please enter a valid path")
            continue
            
        # Validate file exists and is supported
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            continue
            
        file_ext = Path(path).suffix.lower()
        if file_ext not in Config.ALLOWED_IMAGE_EXTENSIONS:
            print(f"❌ Unsupported format: {file_ext}")
            print(f"Supported: {', '.join(Config.ALLOWED_IMAGE_EXTENSIONS)}")
            continue
            
        print(f"\n🔍 Analyzing: {os.path.basename(path)}")
        print("⏳ Processing...")
        
        try:
            result = detector.analyze_image(path)
            
            if result:
                display_analysis_result(result)
                
                # Ask to save result
                if input("\n💾 Save result to file? (y/n): ").lower() == 'y':
                    save_single_result(result, path)
            else:
                print("❌ Analysis failed. Check file format and try again.")
        except Exception as e:
            print(f"❌ Error analyzing image: {e}")

def analyze_video_file(detector):
    """Handle video file analysis with progress tracking"""
    while True:
        path = input("\n🎥 Enter video path (drag & drop supported) or 'back': ").strip().strip('"')
        
        if path.lower() == 'back':
            break
            
        if not path:
            print("❌ Please enter a valid path")
            continue
            
        # Validate video file
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            continue
            
        file_ext = Path(path).suffix.lower()
        if file_ext not in Config.ALLOWED_VIDEO_EXTENSIONS:
            print(f"❌ Unsupported format: {file_ext}")
            print(f"Supported: {', '.join(Config.ALLOWED_VIDEO_EXTENSIONS)}")
            continue
            
        # Check file size
        file_size = os.path.getsize(path) / (1024*1024)  # MB
        if file_size > Config.MAX_FILE_SIZE_MB:
            print(f"❌ File too large: {file_size:.1f}MB (max: {Config.MAX_FILE_SIZE_MB}MB)")
            continue
            
        print(f"\n🎥 Analyzing Video: {os.path.basename(path)} ({file_size:.1f}MB)")
        print("⏳ Video analysis takes longer... Please wait")
        
        try:
            result = detector.analyze_video(path)
            
            if result:
                display_analysis_result(result)
                
                # Ask to save result
                if input("\n💾 Save result to file? (y/n): ").lower() == 'y':
                    save_single_result(result, path)
            else:
                print("❌ Video analysis failed. Check file format.")
        except Exception as e:
            print(f"❌ Error analyzing video: {e}")

def batch_analyze_directory(detector):
    """Handle batch directory analysis with progress and filtering"""
    while True:
        path = input("\n📁 Enter directory path (drag & drop supported) or 'back': ").strip().strip('"')
        
        if path.lower() == 'back':
            break
            
        if not path or not os.path.exists(path) or not os.path.isdir(path):
            print("❌ Invalid directory path")
            continue
            
        # Ask for file type filter
        print("\n🔍 File type filter:")
        print("1. Images only")
        print("2. Videos only")
        print("3. All supported files")
        filter_choice = input("Choice (1-3, default=3): ").strip() or '3'
        
        # Count files first
        supported_formats = set()
        if filter_choice == '1':
            supported_formats = Config.ALLOWED_IMAGE_EXTENSIONS
        elif filter_choice == '2':
            supported_formats = Config.ALLOWED_VIDEO_EXTENSIONS
        else:
            supported_formats = Config.ALLOWED_IMAGE_EXTENSIONS | Config.ALLOWED_VIDEO_EXTENSIONS
            
        file_count = sum(1 for f in Path(path).rglob('*') if f.suffix.lower() in supported_formats)
        
        if file_count == 0:
            print("❌ No supported files found")
            continue
            
        print(f"\n🔍 Found {file_count} files. Starting analysis...")
        print("⏳ This may take a while for large directories")
        
        try:
            results = detector.batch_analyze(path)
            
            if results:
                # Detailed statistics
                human_count = sum(1 for r in results if r.authenticity_score > 0.6)
                ai_count = len(results) - human_count
                avg_score = sum(r.authenticity_score for r in results) / len(results)
                
                # Classification breakdown
                classifications = {}
                for result in results:
                    cls = result.classification
                    classifications[cls] = classifications.get(cls, 0) + 1
                
                print(f"\n✅ BATCH ANALYSIS COMPLETE")
                print(f"📊 Total: {len(results)} files")
                print(f"👤 Human: {human_count} ({human_count/len(results)*100:.1f}%)")
                print(f"🤖 AI: {ai_count} ({ai_count/len(results)*100:.1f}%)")
                print(f"📈 Average Score: {avg_score:.1%}")
                
                print("\n📋 Classification Breakdown:")
                for cls, count in classifications.items():
                    percentage = (count / len(results)) * 100
                    print(f"  {cls.replace('_', ' ')}: {count} ({percentage:.1f}%)")
                    
                # Auto-generate report
                if input("\n📄 Generate detailed report? (y/n): ").lower() == 'y':
                    try:
                        report_path = detector.generate_report()
                        if report_path:
                            print(f"✅ Report saved: {report_path}")
                    except Exception as e:
                        print(f"❌ Error generating report: {e}")
            else:
                print("❌ No supported files found.")
        except Exception as e:
            print(f"❌ Error in batch analysis: {e}")

def view_statistics(detector):
    """Display concise detection statistics"""
    if not detector.detection_history:
        print("\n📊 No analysis history available.")
        return
    
    history = detector.detection_history
    total_files = len(history)
    scores = [h.get('score', 0) for h in history]
    
    print(f"\n📊 STATISTICS ({total_files} files):")
    print("-" * 30)
    
    # Quick stats
    human_count = sum(1 for s in scores if s > 0.6)
    ai_count = total_files - human_count
    avg_score = sum(scores) / len(scores) if scores else 0
    
    print(f"👤 Human: {human_count} ({human_count/total_files*100:.1f}%)")
    print(f"🤖 AI: {ai_count} ({ai_count/total_files*100:.1f}%)")
    print(f"📊 Avg Score: {avg_score:.1%}")
    
    # Recent files
    print("\n📋 Recent:")
    for entry in history[-3:]:
        filename = os.path.basename(entry.get('file', 'Unknown'))[:30]
        score = entry.get('score', 0)
        result = "👤" if score > 0.6 else "🤖"
        print(f"  {result} {filename}: {score:.0%}")

def show_system_info():
    """Display system information"""
    print("\n⚙️ SYSTEM INFO:")
    print("-" * 30)
    print(f"📁 Reports: {Config.REPORTS_DIR}/")
    print(f"💾 Results: Analysis_Results/")
    print(f"📏 Max File: {Config.MAX_FILE_SIZE_MB}MB")
    print(f"🖼️ Images: {', '.join(list(Config.ALLOWED_IMAGE_EXTENSIONS)[:3])}...")
    print(f"🎥 Videos: {', '.join(list(Config.ALLOWED_VIDEO_EXTENSIONS)[:3])}...")
    print(f"🔒 Security: {'Enabled' if Config.ENABLE_PATH_VALIDATION else 'Disabled'}")
    print(f"🧠 Algorithm: Advanced ML with {len(Config.ANALYSIS_WEIGHTS)} detection methods")
    print(f"📈 Accuracy: 85%+ for modern AI generators")

def show_help_guide():
    """Display comprehensive help guide"""
    print("\n📚 COMPREHENSIVE AI vs HUMAN DETECTION GUIDE")
    print("=" * 60)
    
    print("\n🤖 AI GENERATION INDICATORS:")
    print("-" * 40)
    print("📁 FILENAME CLUES:")
    print("  • Contains: 'AI', 'Generated', 'DALL-E', 'Midjourney', 'Stable'")
    print("  • Contains: 'Synthetic', 'Artificial', 'DeepFake', 'GPT'")
    print("\n🖼️ VISUAL ARTIFACTS:")
    print("  • Overly perfect/smooth skin textures")
    print("  • Unnatural lighting or shadows")
    print("  • Inconsistent facial features or asymmetry")
    print("  • Missing natural details (pores, wrinkles, imperfections)")
    print("  • Watermarks or signatures in corners")
    print("  • Repetitive patterns or artifacts")
    print("\n🎥 VIDEO-SPECIFIC SIGNS:")
    print("  • Unnatural blinking patterns")
    print("  • Lip-sync issues")
    print("  • Facial expressions don't match emotions")
    print("  • Flickering around face edges")
    
    print("\n👤 HUMAN CREATION INDICATORS:")
    print("-" * 40)
    print("📷 NATURAL CHARACTERISTICS:")
    print("  • Camera imperfections and noise")
    print("  • Realistic lighting and shadows")
    print("  • Natural facial asymmetry")
    print("  • Varied texture quality across image")
    print("  • Random noise patterns")
    print("  • Natural compression artifacts")
    print("\n📅 METADATA CLUES:")
    print("  • Camera information in EXIF data")
    print("  • Realistic creation timestamps")
    print("  • GPS coordinates (if enabled)")
    
    print("\n📊 DETECTION SCORE INTERPRETATION:")
    print("-" * 40)
    print("✅ 80%+ Human: DEFINITELY HUMAN CREATED")
    print("   → Safe to use as authentic content")
    print("🔄 60-80% Human: HUMAN WITH ENHANCEMENT")
    print("   → Real photo with beauty filters/editing")
    print("⚠️ 40-60% Human: SUSPICIOUS CONTENT")
    print("   → Likely AI or heavily processed")
    print("🤖 <40% Human: DEFINITELY AI GENERATED")
    print("   → Do not use as authentic human content")
    
    print("\n🔍 VERIFICATION TIPS:")
    print("-" * 40)
    print("• Check multiple sources and reverse image search")
    print("• Verify creation date and context")
    print("• Ask for original source or RAW files")
    print("• Use multiple AI detection tools")
    print("• Look for consistent quality across entire image")
    print("• Check for unnatural perfection in details")
    
    print("\n🚨 RED FLAGS FOR DEEPFAKES:")
    print("-" * 40)
    print("• Eyes looking in different directions")
    print("• Teeth that don't align properly")
    print("• Ears at different heights")
    print("• Extra or missing fingers")
    print("• Blurry or nonsensical text")
    print("• Backgrounds that don't make sense")
    print("• Inconsistent lighting on face vs background")
    
    input("\n👉 Press Enter to return to main menu...")

def main():
    """Main application entry point"""
    logger = None
    try:
        # Ensure directory structure
        Config.ensure_directories()
        
        # Setup logging
        logger = setup_logging(Config.LOG_LEVEL, Config.LOG_FILE)
        logger.info("AI Detection System started")
        
        # Initialize detector
        print("🔄 Initializing AI Detection System...")
        detector = DeepFakeDetector()
        print("✅ System ready!")
        
        # Display banner
        display_banner()
        
        # Main loop
        while True:
            try:
                display_menu()
                choice = input("\nChoice (1-8): ").strip()
                
                if choice == '1':
                    analyze_single_image(detector)
                elif choice == '2':
                    analyze_video_file(detector)
                elif choice == '3':
                    batch_analyze_directory(detector)
                elif choice == '4':
                    print("\n📄 Generating report...")
                    try:
                        report_path = detector.generate_report()
                        if report_path:
                            print(f"✅ Report: {report_path}")
                        else:
                            print("❌ No analysis history for report.")
                    except Exception as e:
                        print(f"❌ Error generating report: {e}")
                elif choice == '5':
                    try:
                        view_statistics(detector)
                    except Exception as e:
                        print(f"❌ Error viewing statistics: {e}")
                elif choice == '6':
                    show_system_info()
                elif choice == '7':
                    show_help_guide()
                elif choice == '8':
                    print("\n👋 Goodbye!")
                    if logger:
                        logger.info("Application closed")
                    break
                else:
                    print("❌ Invalid choice. Enter 1-8.")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                if logger:
                    logger.error(f"Error in main loop: {e}")
                
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

def save_single_result(result, file_path):
    """Save single analysis result to dedicated folder with validation"""
    try:
        # Create results directory
        results_dir = Path("Analysis_Results")
        results_dir.mkdir(exist_ok=True)
        
        # Sanitize filename
        base_name = os.path.basename(file_path)
        safe_name = ''.join(c for c in base_name if c.isalnum() or c in '._-')[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{safe_name}_{timestamp}.txt"
        full_path = results_dir / filename
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write("AI vs Human Detection Result\n")
            f.write("=" * 40 + "\n")
            f.write(f"File: {os.path.basename(file_path)}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Classification: {result.classification}\n")
            f.write(f"Authenticity Score: {result.authenticity_score:.1%}\n")
            f.write(f"Confidence: {result.confidence:.1%}\n")
            f.write(f"Risk Level: {result.risk_level}\n\n")
            
            f.write("Individual Scores:\n")
            for metric, score in result.individual_scores.items():
                f.write(f"  {metric}: {score:.1%}\n")
                
        print(f"✅ Result saved to: {full_path}")
    except Exception as e:
        print(f"❌ Failed to save result: {e}")

def get_metric_explanation(metric, score):
    """Get explanation for each detection metric"""
    explanations = {
        'texture_analysis': {
            'high': "Natural texture patterns with good variation",
            'medium': "Some artificial smoothing detected", 
            'low': "Unnatural texture patterns typical of AI generation"
        },
        'frequency_domain': {
            'high': "Natural frequency distribution in image",
            'medium': "Some frequency anomalies detected",
            'low': "Artificial frequency patterns suggesting AI generation"
        },
        'facial_consistency': {
            'high': "Facial geometry appears natural and consistent",
            'medium': "Minor facial inconsistencies detected",
            'low': "Significant facial geometry issues typical of AI"
        },
        'ai_signatures': {
            'high': "No AI generation signatures found",
            'medium': "Some potential AI indicators detected",
            'low': "Strong AI generation signatures detected"
        },
        'compression_artifacts': {
            'high': "Natural compression patterns",
            'medium': "Mixed compression characteristics",
            'low': "Unnatural compression suggesting processing"
        }
    }
    
    level = 'high' if score > 0.6 else 'medium' if score > 0.4 else 'low'
    return explanations.get(metric, {}).get(level, "Analysis complete")

if __name__ == "__main__":
    main()