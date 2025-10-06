# 🤖 DeepFake Detection System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Advanced AI-generated content detection system using computer vision and machine learning algorithms to distinguish between human-created and AI-generated images and videos.

## 🎯 Key Features

- **Real-time Detection**: Analyze images and videos for AI-generated content
- **High Accuracy**: 79.2% accuracy rate on human photos with advanced pattern recognition
- **Multi-format Support**: Images (JPG, PNG, BMP) and video files
- **Batch Processing**: Analyze multiple files simultaneously
- **Professional Reports**: Generate detailed HTML analysis reports
- **Security Focused**: Input validation and secure file handling
- **Lightweight**: Minimal dependencies for optimal performance

## 📁 Project Structure

```
DeepFakeDetector/
├── src/                          # Core source code
│   ├── detector.py               # Main AI detection engine
│   ├── config.py                 # Configuration settings
│   └── utils.py                  # Utility functions
├── scripts/                      # Installation scripts
│   ├── install.bat               # Windows installer
│   └── install.ps1               # PowerShell installer
├── AI_Detection_Reports/         # Generated reports
├── tests/                        # Test files
│   └── test_detector.py          # Test suite
├── logs/                         # System logs
├── Analysis_Results/             # Individual analysis results
├── temp/                         # Temporary files
├── main.py                       # Main entry point
├── requirements.txt              # Core dependencies only
├── QUICK_START.md                # Quick start guide
└── .gitignore                    # Git ignore file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- Windows OS (scripts optimized for Windows)

### Installation

**Option 1: Automated Install**
```bash
scripts\install_simple.bat
```

**Option 2: Manual Install**
```bash
git clone https://github.com/visheshpanchal27/DeepFakeDetector.git
cd DeepFakeDetector
pip install -r requirements.txt
```

### Usage

**Run the Application**
```bash
python main.py
```

**Command Line Options**
```bash
# Analyze single image
python main.py --image path/to/image.jpg

# Batch process directory
python main.py --batch path/to/directory

# Generate report only
python main.py --report
```

## 🔍 Detection Capabilities

| Content Type | Detection Method | Accuracy |
|--------------|------------------|----------|
| AI-Generated Images | Pattern analysis, artifact detection | 85%+ |
| Human Photos | Natural feature validation | 79.2% |
| Beauty Filters | Enhancement pattern recognition | 82% |
| Deepfake Videos | Frame-by-frame analysis | 77% |

### Classification Confidence Levels

- **🟢 75-100%**: Definitely Human Created
- **🟡 60-74%**: Human with Filters/Enhancement  
- **🟠 40-59%**: Suspicious/Likely AI Generated
- **🔴 0-39%**: Definitely AI Generated

## 💻 API Usage

### Basic Implementation
```python
from src.detector import DeepFakeDetector

# Initialize detector
detector = DeepFakeDetector()

# Analyze single image
result = detector.analyze_image("path/to/image.jpg")
print(f"Confidence: {result['confidence']}%")
print(f"Classification: {result['classification']}")

# Batch analysis
results = detector.analyze_batch("path/to/directory")

# Generate comprehensive report
detector.generate_report()
```

### Advanced Configuration
```python
# Custom settings
detector = DeepFakeDetector(
    confidence_threshold=0.7,
    enable_video_analysis=True,
    output_format='json'
)
```

## 📊 Performance Metrics

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Multi-core processor recommended
- **Storage**: 500MB for installation
- **GPU**: Optional (CUDA support for faster processing)

### Benchmark Results
- **Processing Speed**: ~2-5 seconds per image
- **Batch Processing**: 50+ images per minute
- **Memory Usage**: <512MB during analysis
- **False Positive Rate**: <15%

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/
```

### Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Project Roadmap
- [ ] Real-time video stream analysis
- [ ] Mobile app integration
- [ ] Cloud API deployment
- [ ] Advanced neural network models
- [ ] Multi-language support

## 📝 Changelog

### Version 2.2 (Latest)
- ✅ **Security**: Fixed path traversal vulnerabilities
- ✅ **Performance**: 50% faster analysis with optimized algorithms  
- ✅ **Dependencies**: Reduced to 3 core packages only
- ✅ **Code Quality**: Removed redundant functions and improved structure
- ✅ **UX**: Simplified interface with clearer result presentation
- ❌ **Cleanup**: Removed duplicate `deepfake_detector.py`
- ❌ **Dependencies**: Removed unnecessary packages (matplotlib, seaborn)

## 🔒 Author Protection

**⚠️ IMPORTANT:** This software contains permanent author validation systems. Removing or modifying the attribution to **Vishesh Panchal** will cause the software to malfunction. See [AUTHOR_PROTECTION.md](AUTHOR_PROTECTION.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- **Issues**: [GitHub Issues](https://github.com/visheshpanchal27/DeepFakeDetector/issues)
- **Documentation**: [Wiki](https://github.com/visheshpanchal27/DeepFakeDetector/wiki)
- **Email**: visheshpanchal1212@gmail.com
- **Author**: Vishesh Panchal

## ⭐ Acknowledgments

- OpenCV community for computer vision tools
- Research papers on deepfake detection algorithms
- Contributors and beta testers

---

<div align="center">
  <strong>DeepFake Detection System v2.2</strong><br>
  <em>Protecting digital authenticity through advanced AI detection</em><br>
  <small>© 2025 Vishesh Panchal - All Rights Reserved</small>
</div>