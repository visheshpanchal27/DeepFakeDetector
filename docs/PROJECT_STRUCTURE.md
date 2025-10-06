# 📁 DeepFakeDetector Project Structure

```
DeepFakeDetector/
├── 📂 src/                          # Core source code
│   ├── 🐍 improved_detector.py      # Main AI detection engine
│   ├── ⚙️ config.py                 # Configuration settings
│   └── 🛠️ utils.py                  # Utility functions
├── 📂 scripts/                      # Installation scripts
│   ├── 🪟 install.bat               # Windows installer
│   └── 💻 install.ps1               # PowerShell installer
├── 📂 AI_Detection_Reports/         # Generated HTML/JSON reports
├── 📂 Analysis_Results/             # Individual analysis results
├── 📂 logs/                         # System logs
├── 📂 temp/                         # Temporary files
├── 📂 docs/                         # Documentation
├── 📂 examples/                     # Example files
├── 🐍 main.py                       # Main entry point
├── 📋 requirements.txt              # Dependencies
├── 🚀 install_simple.bat            # Quick installer
├── 🧪 test_improved.py              # Test suite
├── 📖 README.md                     # Project documentation
└── 🏃 run.bat                       # Quick run script
```

## 📂 Folder Descriptions

- **src/**: Core detection algorithms and configuration
- **AI_Detection_Reports/**: Comprehensive HTML reports with statistics
- **Analysis_Results/**: Individual file analysis results (TXT format)
- **logs/**: System logs and error tracking
- **temp/**: Temporary processing files
- **scripts/**: Installation and setup scripts
- **docs/**: Additional documentation
- **examples/**: Sample files for testing

## 🎯 File Organization

All output files are automatically organized into appropriate folders:
- Analysis results → `Analysis_Results/`
- Batch reports → `AI_Detection_Reports/`
- System logs → `logs/`
- Temporary files → `temp/`