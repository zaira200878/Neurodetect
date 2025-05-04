import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
from scipy import signal
import winsound  # For playing click sound (Windows only)

class BrainWaveApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroDetect: Brainwave AI Analysis")
        self.setGeometry(100, 100, 1400, 750)
        self.fs = 250
        self.t = np.linspace(0, 2, self.fs * 2)

        self.initUI()
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plot)
        self.running = False

        # Simulated Bluetooth connection status
        self.bluetooth_connected = False

    def initUI(self):
        main_layout = QtWidgets.QHBoxLayout()

        # Left Panel - Controls
        control_panel = QtWidgets.QVBoxLayout()

        title = QtWidgets.QLabel("NeuroDetect - Neural Pattern Detector")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; color: white; background-color: #2c3e50; padding: 10px; border-radius: 10px;")
        control_panel.addWidget(title)

        self.scan_type = QtWidgets.QComboBox()
        self.scan_type.addItems(["Basic Scan", "Deep Scan"])
        self.scan_type.setStyleSheet("font-size: 16px; padding: 5px;")
        control_panel.addWidget(QtWidgets.QLabel("Select Scan Type:"))
        control_panel.addWidget(self.scan_type)

        self.start_btn = QtWidgets.QPushButton("Start Scan")
        self.start_btn.setStyleSheet("padding: 10px; font-size: 16px;")
        self.start_btn.clicked.connect(self.start_scan)
        control_panel.addWidget(self.start_btn)

        self.stop_btn = QtWidgets.QPushButton("Stop Scan")
        self.stop_btn.setStyleSheet("padding: 10px; font-size: 16px;")
        self.stop_btn.clicked.connect(self.stop_scan)
        control_panel.addWidget(self.stop_btn)

        self.analysis_label = QtWidgets.QLabel("Status: Ready to scan")
        self.analysis_label.setWordWrap(True)
        self.analysis_label.setStyleSheet("font-size: 16px; color: white; background-color: #34495e; padding: 10px; border-radius: 10px;")
        control_panel.addWidget(self.analysis_label)

        control_panel.addWidget(QtWidgets.QLabel("\nAI Analysis:"))
        self.ai_result = QtWidgets.QTextEdit()
        self.ai_result.setReadOnly(True)
        self.ai_result.setStyleSheet("font-size: 16px; color: #000000; background-color: #ffffff; padding: 10px; border: 1px solid #ccc;")
        control_panel.addWidget(self.ai_result)

        control_panel.addWidget(QtWidgets.QLabel("\nMedical Suggestion:"))
        self.medical_suggestion = QtWidgets.QTextEdit()
        self.medical_suggestion.setReadOnly(True)
        self.medical_suggestion.setStyleSheet("font-size: 16px; color: #000000; background-color: #e8f8f5; padding: 10px; border: 1px solid #ccc;")
        control_panel.addWidget(self.medical_suggestion)

        control_panel.addWidget(QtWidgets.QLabel("\nLifestyle Recommendation:"))
        self.lifestyle_recommendation = QtWidgets.QTextEdit()
        self.lifestyle_recommendation.setReadOnly(True)
        self.lifestyle_recommendation.setStyleSheet("font-size: 16px; color: #000000; background-color: #f5f5f5; padding: 10px; border: 1px solid #ccc;")
        control_panel.addWidget(self.lifestyle_recommendation)

        # Bluetooth connection button moved here
        self.bluetooth_btn = QtWidgets.QPushButton("Connect Bluetooth EEG")
        self.bluetooth_btn.setStyleSheet("padding: 12px; font-size: 18px; background-color: #1abc9c; color: white; border-radius: 8px;")
        self.bluetooth_btn.clicked.connect(self.toggle_bluetooth)
        control_panel.addWidget(self.bluetooth_btn)

        control_panel.addStretch()

        # Right Panel - EEG Graph
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setYRange(-3, 3)
        self.plot_widget.setBackground('#1c2833')
        self.curve = self.plot_widget.plot(pen=pg.mkPen('cyan', width=2))

        # Layout adjustments
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.plot_widget)

        main_layout.addLayout(control_panel, 4)
        main_layout.addLayout(right_layout, 6)

        self.setLayout(main_layout)
        self.setStyleSheet("background-color: #ecf0f1;")

    def toggle_bluetooth(self):
        # Play click sound effect
        winsound.Beep(1000, 200)  # Windows sound (can be replaced with other sound libraries)

        if self.bluetooth_connected:
            self.bluetooth_connected = False
            self.bluetooth_btn.setText("Connect Bluetooth EEG")
            self.analysis_label.setText("Status: Bluetooth disconnected")
        else:
            self.bluetooth_connected = True
            self.bluetooth_btn.setText("Disconnect Bluetooth EEG")
            self.analysis_label.setText("Status: Bluetooth connected")

    def start_scan(self):
        self.running = True
        self.timer.start()
        self.analysis_label.setText("Status: Scanning... please wait")
        self.ai_result.clear()
        self.medical_suggestion.clear()
        self.lifestyle_recommendation.clear()

    def stop_scan(self):
        self.running = False
        self.timer.stop()
        self.analysis_label.setText("Status: Scan stopped")

    def generate_eeg(self):
        alpha = np.sin(2 * np.pi * 10 * self.t)
        beta = 0.5 * np.sin(2 * np.pi * 20 * self.t)
        gamma = 0.3 * np.sin(2 * np.pi * 40 * self.t)
        noise = np.random.normal(0, 0.5, len(self.t))
        return alpha + beta + gamma + noise

    def analyze_brain(self, signal_data):
        b, a = signal.butter(4, [8 / (0.5 * self.fs), 12 / (0.5 * self.fs)], btype='band')
        filtered = signal.filtfilt(b, a, signal_data)
        power = np.mean(filtered ** 2)

        scan_mode = self.scan_type.currentText()

        if power > 0.8:
            summary = "Normal brain activity."
            diseases = ["No signs of Alzheimer's", "No signs of Parkinson's", "Cognitive function normal"]
            medical_suggestion = "No immediate concerns. Keep up with regular checkups and maintain overall well-being."
            lifestyle_recommendation = """
            - Maintain a balanced diet with brain-boosting nutrients.
            - Exercise regularly to support cognitive health.
            - Engage in mentally stimulating activities, such as reading or puzzles.
            - Get adequate sleep each night to refresh your brain.
            """
        elif power > 0.5:
            summary = "Mild neural irregularities detected."
            diseases = ["Early risk of Alzheimer's", "Monitor for stress-related disorders"]
            medical_suggestion = "It is recommended to consult with a neurologist for further testing. Monitor your mental health regularly."
            lifestyle_recommendation = """
            - Try stress management techniques such as meditation or yoga.
            - Sleep improvement can have a positive impact on cognitive health.
            - Engage in mindfulness exercises to reduce mental fatigue.
            """
        elif power > 0.3:
            summary = "Possible early-stage neurological symptoms."
            diseases = ["Early signs of Dementia", "Increased risk of Parkinson's disease"]
            medical_suggestion = "Schedule an MRI and neuro evaluation. Consider lifestyle changes to slow progression."
            lifestyle_recommendation = """
            - Cognitive exercises to improve memory retention and brain function.
            - Schedule a visit to a neurologist for a more thorough evaluation.
            - Ensure you're eating a brain-healthy diet with antioxidants.
            """
        else:
            summary = "Low neural activity detected."
            diseases = ["High risk of Alzheimer's or related disorders", "Immediate attention required"]
            medical_suggestion = "Urgent consultation with a neurologist is recommended for a full evaluation."
            lifestyle_recommendation = """
            - Seek professional guidance immediately.
            - Engage in cognitive rehabilitation exercises.
            - Ensure proper sleep, exercise, and stress management techniques.
            """

        disease_details = "\n\nPossible Early Disease Detections:\n- " + "\n- ".join(diseases)
        detail = f"Scan Mode: {scan_mode}\nAlpha Wave Power: {power:.2f}\n\nInterpretation:\n{summary}{disease_details}"
        
        # Set the "Medical Suggestion" and "Lifestyle Recommendation" in the UI
        self.medical_suggestion.setText(medical_suggestion)
        self.lifestyle_recommendation.setText(lifestyle_recommendation)

        return summary, detail, medical_suggestion

    def update_plot(self):
        if not self.running:
            return

        eeg_signal = self.generate_eeg()
        self.curve.setData(eeg_signal)

        summary, detail, medical_suggestion = self.analyze_brain(eeg_signal)
        self.analysis_label.setText(f"Status: {summary}")
        self.ai_result.setText(detail)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = BrainWaveApp()
    win.show()
    sys.exit(app.exec_())
