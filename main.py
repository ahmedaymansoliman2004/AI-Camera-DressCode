import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.main_window import MainWindow
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())