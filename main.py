#!/usr/bin/env python3.10
"""
AI Dress Code Detector - Main Entry Point
This script launches the PyQt5 GUI application.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
# This ensures the 'app' package can be imported correctly
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def main():
    """Main function to initialize and run the application."""
    try:
        # Import and initialize Qt Application
        from PyQt5.QtWidgets import QApplication
        from app.main_window import MainWindow
        
        # Create the QApplication instance (required for PyQt)
        app = QApplication(sys.argv)
        app.setApplicationName("AI Dress Code Monitor")
        app.setApplicationVersion("1.0.0")
        
        # Create and show the main window
        window = MainWindow()
        window.show()
        
        # Start the application event loop
        print("Application started successfully. Press Ctrl+C to exit.")
        return_code = app.exec_()
        
        # Clean exit
        sys.exit(return_code)
        
    except ImportError as e:
        print(f"Error: Missing dependency. {e}")
        print("Please install the required packages:")
        print("pip3.10 install PyQt5 picamera2 opencv-python-headless onnxruntime")
        sys.exit(1)
        
    except Exception as e:
        print(f"Fatal error during application startup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
