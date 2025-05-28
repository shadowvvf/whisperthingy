# 🔊 Whisper Desktop Transcriber

This is a simple desktop application that helps you convert speech into text using OpenAI's powerful Whisper model. You can record audio directly from your microphone or open existing audio files, and then quickly get their transcription.

## ✨ What the Program Does:

*   **🎤 Voice Recording:** Record audio directly from your microphone.
*   **📁 File Opening:** Open your audio files (MP3, WAV, FLAC, OGG) for transcription.
*   **📝 Transcription:** Convert recorded or opened speech into text.
*   **⚙️ Whisper Settings:**
    *   **Model Selection:** Use different Whisper models (from "tiny" to "large" and "turbo") to balance speed and accuracy.
    *   **Language Selection:** Specify the language of the speech (many languages are supported!).
    *   **Device Selection:** Choose whether to use your CPU or GPU (CUDA) for transcription.
    *   **File Cleanup:** Automatically delete temporary transcription files (.txt, .srt, etc.) after getting the text.
*   **💾 Text Saving:** Save the generated text to a plain text file (`.txt`).
*   **🧹 Clear:** Easily clear the text output field.

## 🚀 How to Get Started:

### 🛠️ What You'll Need:

1.  **Python 3.x:** Make sure you have Python installed (version 3.8 or newer is recommended).
2.  **`ffmpeg`:** This is a tool for working with audio. Install it using your system's package manager:
    *   **Windows (Chocolatey):** `choco install ffmpeg`
    *   **Windows (Scoop):** `scoop install ffmpeg`
    *   **macOS (Homebrew):** `brew install ffmpeg`
    *   **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
    *   **Arch Linux:** `sudo pacman -S ffmpeg`

### 📦 Installation:

1.  **Download the Code:** Get this project onto your computer (e.g., by clicking "Code" -> "Download ZIP" or cloning the repository).
2.  **Open Terminal/Command Prompt:** Navigate to the folder where you downloaded the project.
3.  **Create a Virtual Environment (recommended):**
    ```bash
    python -m venv venv
    ```
4.  **Activate the Virtual Environment:**
    *   **Windows:** `.\venv\Scripts\activate`
    *   **macOS/Linux:** `source venv/bin/activate`
5.  **Install Required Libraries:**
    ```bash
    pip install PyQt5 PyAudio openai-whisper
    ```
    *   If you encounter errors during `openai-whisper` installation (e.g., related to `tiktoken`), you might need to install `rust` and `setuptools-rust`:
        ```bash
        pip install setuptools-rust
        ```
        Then retry `pip install openai-whisper`.
6.  **(Optional) Add an Icon:** If you want the application to have an icon, place an `icon.png` file (preferably 256x256 pixels) in the same folder as `main.py`.

## 🏃‍♀️ How to Use:

1.  **Run the Program:**
    ```bash
    python main.py
    ```
2.  **Recording:**
    *   Click "🎤 Start Recording" to begin recording from your microphone.
    *   Click "⏹️ Stop Recording" to stop recording and save the file.
3.  **Opening a File:**
    *   Click "📁 Open Audio File" and select an audio file from your computer.
4.  **Transcription:**
    *   After recording or opening a file, click "📝 Transcribe". The text will appear in the large text field.
5.  **Settings:**
    *   Use the dropdowns at the top of the window to select the Whisper model, language, and device (CPU/GPU).
    *   Check or uncheck "Keep output files" to control the saving of temporary transcription files.
6.  **Save/Clear:**
    *   Click "💾 Save Text" to save the generated text to a file.
    *   Click "🗑️ Clear" to clear the text field.