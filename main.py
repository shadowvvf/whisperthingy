import sys
import os
import subprocess
import threading
import time
from datetime import datetime
import pyaudio
import wave
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QWidget, QPushButton, QLabel, QTextEdit, QComboBox,
                            QProgressBar, QFrame, QMessageBox, QFileDialog,
                            QCheckBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon

COMMON_WHISPER_LANGUAGES = [
    "Auto", "English", "Russian", "Chinese", "German", "Spanish", "Korean",
    "French", "Japanese", "Portuguese", "Turkish", "Polish", "Catalan",
    "Dutch", "Arabic", "Swedish", "Italian", "Indonesian", "Hindi",
    "Finnish", "Vietnamese", "Hebrew", "Ukrainian", "Greek", "Thai",
    "Czech", "Romanian", "Danish", "Hungarian", "Norwegian", "Urdu",
    "Croatian", "Bulgarian", "Serbian", "Gujarati", "Telugu", "Kannada",
    "Malayalam", "Marathi", "Nepali", "Mongolian", "Bosnian", "Kazakh",
    "Albanian", "Swahili", "Slovenian", "Armenian", "Estonian", "Welsh",
    "Latvian", "Lithuanian", "Macedonian", "Georgian", "Azerbaijani",
    "Afrikaans", "Irish", "Scottish Gaelic", "Luxembourgish", "Yiddish",
    "Icelandic", "Haitian Creole", "Malagasy", "Esperanto", "Persian", "Sanskrit",
    "Lao", "Tibetan", "Burmese", "Tagalog", "Khmer", "Maori", "Sindhi", "Amharic"
]

WHISPER_MODELS = {
    "tiny":      {"params": "39 M",   "en_only": True,  "multilingual": True, "vram": "~1 GB",  "speed": "~10x"},
    "tiny.en":   {"params": "39 M",   "en_only": True,  "multilingual": False,"vram": "~1 GB",  "speed": "~10x"},
    "base":      {"params": "74 M",   "en_only": True,  "multilingual": True, "vram": "~1 GB",  "speed": "~7x"},
    "base.en":   {"params": "74 M",   "en_only": True,  "multilingual": False,"vram": "~1 GB",  "speed": "~7x"},
    "small":     {"params": "244 M",  "en_only": True,  "multilingual": True, "vram": "~2 GB",  "speed": "~4x"},
    "small.en":  {"params": "244 M",  "en_only": True,  "multilingual": False,"vram": "~2 GB",  "speed": "~4x"},
    "medium":    {"params": "769 M",  "en_only": True,  "multilingual": True, "vram": "~5 GB",  "speed": "~2x"},
    "medium.en": {"params": "769 M",  "en_only": True,  "multilingual": False,"vram": "~5 GB",  "speed": "~2x"},
    "large":     {"params": "1550 M", "en_only": False, "multilingual": True, "vram": "~10 GB", "speed": "1x"},
    "large-v2":  {"params": "1550 M", "en_only": False, "multilingual": True, "vram": "~10 GB", "speed": "1x"},
    "large-v3":  {"params": "1550 M", "en_only": False, "multilingual": True, "vram": "~10 GB", "speed": "1x"},
    "turbo":     {"params": "809 M",  "en_only": False, "multilingual": True, "vram": "~6 GB",  "speed": "~8x (optimized large-v3)"}
}

class TranscriptionWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress_update = pyqtSignal(str)

    def __init__(self, audio_file, language, model_name, device, keep_output_files):
        super().__init__()
        self.audio_file = audio_file
        self.language = language
        self.model_name = model_name
        self.device = device
        self.keep_output_files = keep_output_files
        self.whisper_executable = self._find_whisper_executable()
        self._is_error_emitted = False

    def _find_whisper_executable(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        venv_rel_paths = [
            os.path.join(script_dir, "venv"),
            os.path.join(script_dir, "..", "venv"),
        ]

        exec_name = "whisper.exe" if sys.platform == "win32" else "whisper"

        for venv_path in venv_rel_paths:
            if sys.platform == "win32":
                venv_exec_path = os.path.join(venv_path, "Scripts", exec_name)
            else:
                venv_exec_path = os.path.join(venv_path, "bin", exec_name)
            
            if os.path.exists(venv_exec_path):
                return venv_exec_path
        
        try:
            subprocess.run([exec_name, '--version'], capture_output=True, text=True, check=True)
            return exec_name
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

    def run(self):
        if not self.whisper_executable:
            self.error.emit(f"Whisper executable not found. Please ensure 'whisper' is installed and accessible in your environment (virtual environment or PATH).")
            self._is_error_emitted = True
            return

        try:
            self.progress_update.emit("Preparing transcription command...")
            
            cmd = [self.whisper_executable, self.audio_file, "--model", self.model_name]
            
            if self.language and self.language != "Auto":
                cmd.extend(["--language", self.language])
            
            if self.device == "GPU (CUDA)":
                cmd.extend(["--device", "cuda"])
            elif self.device == "CPU":
                cmd.extend(["--device", "cpu"])

            output_dir = os.path.dirname(os.path.abspath(self.audio_file))
            cmd.extend(["--output_dir", output_dir, "--output_format", "txt"])

            self.progress_update.emit(f"Starting transcription (model: '{self.model_name}', device: '{self.device}')...")
            
            work_dir = os.path.dirname(os.path.abspath(self.audio_file))
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=work_dir, bufsize=1)
            
            stdout_thread = threading.Thread(target=self._read_stream, args=(process.stdout, "stdout"))
            stderr_thread = threading.Thread(target=self._read_stream, args=(process.stderr, "stderr"))
            stdout_thread.start()
            stderr_thread.start()
            
            process.wait()
            stdout_thread.join()
            stderr_thread.join()
            
            if process.returncode == 0:
                base_name = os.path.splitext(os.path.basename(self.audio_file))[0]
                txt_file = os.path.join(output_dir, base_name + ".txt")

                if os.path.exists(txt_file):
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        transcription = f.read().strip()
                    
                    if not self.keep_output_files:
                        self.cleanup_temp_files(os.path.join(output_dir, base_name))
                    else:
                        self.progress_update.emit(f"Transcription files saved to: {output_dir}")
                    
                    self.finished.emit(transcription)
                else:
                    error_message = f"Transcription file not found at: {txt_file}\n" \
                                    f"Please ensure Whisper successfully created the file."
                    if not self._is_error_emitted:
                        self.error.emit(error_message)
                        self._is_error_emitted = True
            else:
                error_message = f"Whisper process exited with error code: {process.returncode}.\n" \
                                f"Check console for more details."
                if not self._is_error_emitted:
                    self.error.emit(error_message)
                    self._is_error_emitted = True
        except Exception as e:
            if not self._is_error_emitted:
                self.error.emit(f"Transcription error: {str(e)}")
                self._is_error_emitted = True

    def _read_stream(self, pipe, stream_type):
        for line in iter(pipe.readline, ''):
            clean_line = line.strip()
            if clean_line:
                if stream_type == "stdout":
                    if "Detected language" in clean_line or "Loading audio" in clean_line or "Applying ASR" in clean_line:
                        self.progress_update.emit(clean_line)
                elif stream_type == "stderr":
                    if "error" in clean_line.lower() or "exception" in clean_line.lower() or "failed" in clean_line.lower():
                        if not self._is_error_emitted:
                            self.error.emit(f"Whisper CLI Runtime Error: {clean_line}")
                            self._is_error_emitted = True
                    else:
                        self.progress_update.emit(clean_line)

    def cleanup_temp_files(self, base_path_no_ext):
        extensions_to_remove = ['.txt', '.json', '.srt', '.tsv', '.vtt']
        for ext in extensions_to_remove:
            temp_file = base_path_no_ext + ext
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass


class VoiceRecorderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recording = False
        self.audio_frames = []
        self.audio_stream = None
        self.audio = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_recording_time)
        self.recording_time = 0
        self.transcription_worker = None
        self.last_audio_file = None

        self.init_ui()
        self.setup_audio()

    def init_ui(self):
        self.setWindowTitle("Whisper Desktop Transcriber")
        self.setWindowIcon(QIcon(self.get_icon_path()))
        self.setGeometry(200, 200, 800, 850)
        self.setStyleSheet(self.get_stylesheet())

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(25, 25, 25, 25)

        title = QLabel("🗣️ Whisper Desktop Transcriber")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #2c3e50; margin-bottom: 15px;")
        main_layout.addWidget(title)

        settings_frame = QFrame()
        settings_frame.setStyleSheet("QFrame { background-color: #f0f4f8; border-radius: 12px; padding: 20px; border: 1px solid #dcdcdc; }")
        settings_layout = QGridLayout(settings_frame)
        settings_layout.setSpacing(10)

        settings_layout.addWidget(self.create_label("Model:"), 0, 0, Qt.AlignLeft)
        self.model_combo = self.create_combo_box(list(WHISPER_MODELS.keys()))
        self.model_combo.setCurrentText("turbo")
        self.model_combo.currentIndexChanged.connect(self.update_model_info)
        settings_layout.addWidget(self.model_combo, 0, 1, 1, 2)
        self.model_info_label = self.create_label("")
        self.model_info_label.setStyleSheet("font-size: 11px; color: #555;")
        settings_layout.addWidget(self.model_info_label, 0, 3, 1, 2)
        self.update_model_info()

        settings_layout.addWidget(self.create_label("Language:"), 1, 0, Qt.AlignLeft)
        self.language_combo = self.create_combo_box(COMMON_WHISPER_LANGUAGES)
        self.language_combo.setCurrentText("Auto")
        settings_layout.addWidget(self.language_combo, 1, 1, 1, 4)

        settings_layout.addWidget(self.create_label("Device:"), 2, 0, Qt.AlignLeft)
        self.device_combo = self.create_combo_box(["Auto", "GPU (CUDA)", "CPU"])
        self.device_combo.setCurrentText("Auto")
        settings_layout.addWidget(self.device_combo, 2, 1, 1, 4)

        settings_layout.addWidget(self.create_label("Keep output files (.txt, .srt etc.):"), 3, 0, Qt.AlignLeft)
        self.keep_output_files_checkbox = QCheckBox("")
        self.keep_output_files_checkbox.setChecked(False)
        settings_layout.addWidget(self.keep_output_files_checkbox, 3, 1, Qt.AlignLeft)
        
        main_layout.addWidget(settings_frame)

        self.status_label = QLabel("Ready to record")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 17px; color: #27ae60; font-weight: bold; margin-top: 10px;")
        main_layout.addWidget(self.status_label)

        self.time_label = QLabel("00:00")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 28px; color: #34495e; font-family: 'monospace'; margin-bottom: 10px;")
        main_layout.addWidget(self.time_label)

        control_buttons_layout = QHBoxLayout()
        self.record_button = self.create_button("🎤 Start Recording", self.toggle_recording, "#28a745")
        self.open_file_button = self.create_button("📁 Open Audio File", self.open_audio_file_and_transcribe, "#007bff")
        self.transcribe_button = self.create_button("📝 Transcribe", self.transcribe_audio, "#17a2b8")
        self.transcribe_button.setEnabled(False)

        control_buttons_layout.addWidget(self.record_button)
        control_buttons_layout.addWidget(self.open_file_button)
        control_buttons_layout.addWidget(self.transcribe_button)
        main_layout.addLayout(control_buttons_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ced4da;
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x0:0, y0:0, x1:1, y1:0, stop:0 #17a2b8, stop:1 #007bff);
                border-radius: 8px;
            }
        """)
        main_layout.addWidget(self.progress_bar)

        result_label = QLabel("Transcription Result:")
        result_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #2c3e50; margin-top: 15px;")
        main_layout.addWidget(result_label)

        self.result_text = QTextEdit()
        self.result_text.setPlaceholderText("Transcription text will appear here. You can record or open an existing audio file.")
        self.result_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ced4da;
                border-radius: 8px;
                padding: 15px;
                font-size: 14px;
                background-color: white;
                line-height: 1.5;
            }
            QTextEdit:focus {
                border-color: #007bff;
            }
        """)
        main_layout.addWidget(self.result_text)

        text_buttons_layout = QHBoxLayout()
        self.save_text_button = self.create_button("💾 Save Text", self.save_text, "#6c757d")
        self.clear_button = self.create_button("🗑️ Clear", self.clear_text, "#dc3545")
        
        text_buttons_layout.addWidget(self.save_text_button)
        text_buttons_layout.addWidget(self.clear_button)
        text_buttons_layout.addStretch()
        main_layout.addLayout(text_buttons_layout)
        
        main_layout.addStretch()

    def get_icon_path(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "icon.png")
        if os.path.exists(icon_path):
            return icon_path
        return ""

    def create_label(self, text):
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold; color: #34495e;")
        return label

    def create_combo_box(self, items):
        combo = QComboBox()
        combo.addItems(items)
        combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 5px;
                background-color: white;
                selection-background-color: #007bff;
                selection-color: white;
                color: #34495e;
            }
            QComboBox::drop-down {
                border: 0px;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAZ0lEQVQ4T2NkgAC/gcA/E4A/I4B/O2B/I4A/I4B/I4B/I4B/A2B/I4A/E4A/I4A/I4A/YxggB04D6BqGgWIoGgGIoGgGIoGgGIoGgGIoGgGIoGgGIoGgGAAAAAElFTkSuQmCC); 
                width: 10px;
                height: 10px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #ced4da;
                border-radius: 5px;
                background-color: white;
                selection-background-color: #007bff;
                color: #34495e;
            }
        """)
        return combo

    def _get_button_stylesheet(self, background_color):
        """Generates the stylesheet string for a button."""
        return f"""
            QPushButton {{
                background-color: {background_color};
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 20px;
                font-size: 15px;
                font-weight: bold;
                min-width: 150px;
                transition: background-color 0.2s ease;
            }}
            QPushButton:hover {{
                background-color: {self.darken_color(background_color, 0.1)};
            }}
            QPushButton:pressed {{
                background-color: {self.darken_color(background_color, 0.2)};
            }}
            QPushButton:disabled {{
                background-color: #bdc3c7;
                color: #6c757d;
            }}
        """

    def create_button(self, text, handler, background_color):
        button = QPushButton(text)
        button.clicked.connect(handler)
        button.setStyleSheet(self._get_button_stylesheet(background_color)) # Use the new helper
        return button

    def darken_color(self, hex_color, factor):
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darkened_rgb = tuple(max(0, int(c * (1 - factor))) for c in rgb)
        return '#%02x%02x%02x' % darkened_rgb

    def get_stylesheet(self):
        return """
            QMainWindow {
                background-color: #ecf0f1;
            }
            QLabel {
                color: #2c3e50;
            }
        """

    def update_model_info(self):
        model_name = self.model_combo.currentText()
        info = WHISPER_MODELS.get(model_name, {})
        if info:
            text = f"Params: {info['params']} | VRAM: {info['vram']} | Speed: {info['speed']}"
            if not info['multilingual']:
                text += " (English-only)"
            self.model_info_label.setText(text)
        else:
            self.model_info_label.setText("Model information not available.")

    def setup_audio(self):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        try:
            self.audio = pyaudio.PyAudio()
        except Exception as e:
            QMessageBox.critical(self, "Audio Error", f"Failed to initialize audio system: {str(e)}\n"
                                                      "Please ensure microphone is connected and drivers are installed.")
            self.record_button.setEnabled(False)

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not self.audio:
            QMessageBox.warning(self, "Recording Not Possible", "Audio system not ready. Cannot start recording.")
            return

        try:
            self.audio_frames = []
            self.audio_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            self.recording = True
            self.recording_time = 0
            self.timer.start(1000)
            
            # Apply stop button style using the helper
            self.record_button.setText("⏹️ Stop Recording")
            self.record_button.setStyleSheet(self._get_button_stylesheet("#dc3545"))
            self.status_label.setText("🔴 Recording in progress...")
            self.status_label.setStyleSheet("font-size: 17px; color: #dc3545; font-weight: bold;")
            
            self.transcribe_button.setEnabled(False)
            self.open_file_button.setEnabled(False)
            self.model_combo.setEnabled(False)
            self.language_combo.setEnabled(False)
            self.device_combo.setEnabled(False)
            self.keep_output_files_checkbox.setEnabled(False)

            self.recording_thread = threading.Thread(target=self.record_audio_thread)
            self.recording_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "Recording Error", f"Failed to start recording: {str(e)}\n"
                                                          "Check microphone permissions or if it's in use by another application.")
            self.stop_recording_ui_reset()

    def record_audio_thread(self):
        while self.recording:
            try:
                data = self.audio_stream.read(self.chunk)
                self.audio_frames.append(data)
            except Exception:
                self.recording = False
                break

    def stop_recording(self):
        self.recording = False
        self.timer.stop()
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None

        if not self.audio_frames:
            QMessageBox.warning(self, "No Audio Recorded", "Audio frames were not recorded. Recording might have been too short or interrupted.")
            self.stop_recording_ui_reset(success=False)
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.audio_frames))
            
            self.last_audio_file = filename
            self.stop_recording_ui_reset(success=True, filename=filename)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save audio file: {str(e)}")
            self.stop_recording_ui_reset(success=False)

    def stop_recording_ui_reset(self, success=True, filename=None):
        self.record_button.setText("🎤 Start Recording")
        self.record_button.setStyleSheet(self._get_button_stylesheet("#28a745")) # Use the new helper
        self.transcribe_button.setEnabled(True if success else False)
        self.open_file_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.language_combo.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.keep_output_files_checkbox.setEnabled(True)

        if success and filename:
            self.status_label.setText(f"✅ Recording saved: {os.path.basename(filename)}")
            self.status_label.setStyleSheet("font-size: 17px; color: #27ae60; font-weight: bold;")
        else:
            self.status_label.setText("Ready to record")
            self.status_label.setStyleSheet("font-size: 17px; color: #27ae60; font-weight: bold;")

    def update_recording_time(self):
        self.recording_time += 1
        minutes = self.recording_time // 60
        seconds = self.recording_time % 60
        self.time_label.setText(f"{minutes:02d}:{seconds:02d}")

    def open_audio_file_and_transcribe(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Open Audio File for Transcription")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Audio files (*.wav *.mp3 *.flac *.ogg);;All files (*.*)")
        
        if file_dialog.exec_():
            selected_file = file_dialog.selectedFiles()[0]
            self.last_audio_file = selected_file
            self.transcribe_audio()

    def transcribe_audio(self):
        if not self.last_audio_file or not os.path.exists(self.last_audio_file):
            QMessageBox.warning(self, "Warning", "No audio file to transcribe. Please record or open a file.")
            return
        
        self.transcribe_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.open_file_button.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.language_combo.setEnabled(False)
        self.device_combo.setEnabled(False)
        self.keep_output_files_checkbox.setEnabled(False)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        language = self.language_combo.currentText()
        model_name = self.model_combo.currentText()
        device = self.device_combo.currentText()
        keep_output_files = self.keep_output_files_checkbox.isChecked()

        device_text = "auto" if device == "Auto" else device
        self.status_label.setText(f"🔄 Transcribing ({model_name} / {device_text})...")
        self.status_label.setStyleSheet("font-size: 17px; color: #f39c12; font-weight: bold;")
        self.result_text.setPlaceholderText("Transcription in progress...")

        self.transcription_worker = TranscriptionWorker(
            self.last_audio_file, language, model_name, device, keep_output_files
        )
        self.transcription_worker.finished.connect(self.on_transcription_finished)
        self.transcription_worker.error.connect(self.on_transcription_error)
        self.transcription_worker.progress_update.connect(self.update_status_label)
        self.transcription_worker.start()

    def update_status_label(self, message):
        self.status_label.setText(f"🔄 {message}")
        self.status_label.setStyleSheet("font-size: 17px; color: #f39c12; font-weight: bold;")

    def on_transcription_finished(self, text):
        self.progress_bar.setVisible(False)
        
        self.transcribe_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.open_file_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.language_combo.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.keep_output_files_checkbox.setEnabled(True)

        self.status_label.setText("✅ Transcription complete")
        self.status_label.setStyleSheet("font-size: 17px; color: #27ae60; font-weight: bold;")
        self.result_text.setText(text)

    def on_transcription_error(self, error):
        self.progress_bar.setVisible(False)
        
        self.transcribe_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.open_file_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.language_combo.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.keep_output_files_checkbox.setEnabled(True)

        self.status_label.setText("❌ Transcription error")
        self.status_label.setStyleSheet("font-size: 17px; color: #dc3545; font-weight: bold;")
        QMessageBox.critical(self, "Transcription Error", error)
        self.result_text.setText(f"Error: {error}")

    def save_text(self):
        text = self.result_text.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "Warning", "No text to save.")
            return
        
        suggested_filename = "transcription.txt"
        if self.last_audio_file:
            base_name = os.path.splitext(os.path.basename(self.last_audio_file))[0]
            suggested_filename = f"{base_name}_transcribed.txt"
        else:
            suggested_filename = f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Transcription", suggested_filename,
            "Text files (*.txt);;All files (*.*)"
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                QMessageBox.information(self, "Success", f"Text saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save file: {str(e)}")

    def clear_text(self):
        self.result_text.clear()
        self.result_text.setPlaceholderText("Transcription text will appear here. You can record or open an existing audio file.")

    def closeEvent(self, event):
        if self.recording:
            self.stop_recording()
        if self.transcription_worker and self.transcription_worker.isRunning():
            self.transcription_worker.terminate()
            self.transcription_worker.wait()
        if self.audio:
            self.audio.terminate()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = VoiceRecorderApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()