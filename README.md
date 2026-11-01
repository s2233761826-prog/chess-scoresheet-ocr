# Handwritten Chess Scoresheet → PGN

## What it does

Convert photographs of handwritten chess scoresheets into PGN files for replay and analysis. The app recognizes moves, suggests corrections based on OCR results and chess rules, and lets you review and edit the moves before exporting.

## Installation

Install **Python 3.11**, then download this repository using **Code → Download ZIP** and extract it.

### macOS

Open Terminal, type `bash ` (including the space), drag `start_mac.command` from the extracted folder into Terminal, and press Return.

The first launch installs the required dependencies. When startup finishes, open **http://127.0.0.1:8501** in your browser. Keep Terminal open while using the app.

### Windows

Open PowerShell in the extracted project folder and run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app/main.py
```

### Linux

Open Terminal in the extracted project folder and run:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app/main.py
```

The app has been tested on an Apple Silicon Mac. Windows and Linux setup may require platform-specific dependency adjustments. No GPU or API key is required; the first installation requires internet access.

## How to use

1. Upload a clear JPG or PNG photograph of a scoresheet. Recognition starts automatically.
2. Review the recognized moves alongside the original image. If table detection fails, use the manual alignment controls.
3. Click a suggested correction or edit the move table directly. The app checks the updated moves against chess rules.
4. Check the complete game against the photo, confirm your review, and download the **PGN** file.

Sample files are included in `examples/`. Handwritten moves may be misread, so review is required before export.

To reopen the app on macOS, run `start_mac.command` again using the same Terminal method. On Windows or Linux, run the final command from the corresponding installation section. Press **Control+C** in Terminal to stop the app.
