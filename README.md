# Handwritten Chess Scoresheet → PGN

A local web application that helps chess players digitize handwritten scoresheets. Upload a photograph, review the recognized moves with image-based and chess-rule-based suggestions, and export a PGN file for replay or analysis.

**Status:** working prototype with human review. Handwriting and table recognition can make mistakes; this is not an unattended, guaranteed-accurate transcription service.

## What it does

- Reads supported scoresheet tables with repeated move-number / White / Black columns.
- Shows the original image alongside an editable move table.
- Offers up to four legal move candidates using OCR character probabilities and text similarity.
- Lets the user accept a candidate or type a correction; revalidates after changes.
- Exports PGN after legal validation and explicit user review.

Legal moves are not necessarily the moves written on the paper. The application does not silently rewrite an entire game to make it legal. If an earlier move is wrong but legal, it can affect later suggestions. Paper records can also contain mistakes.

## Run on your computer

This GitHub repository contains code and model files, not a hosted demo. You do not need a GPU or API key. First-time dependency installation requires an internet connection.

### macOS — tested by the project owner on Apple Silicon

1. Install Python 3.11.
2. Download this repository using **Code → Download ZIP**, then extract it.
3. Open Terminal, type `bash ` (with a space), drag `start_mac.command` from the extracted project into Terminal, and press Return.
4. Wait for dependency installation on the first launch. Open the URL printed in Terminal (normally http://127.0.0.1:8501).
5. Keep Terminal running while using the application. Press Control+C to stop it.

For subsequent launches, repeat step 3. The launcher reuses the local environment. PyCharm is optional.

### Windows / Linux — setup instructions, not end-to-end verified in this release

Use 64-bit Python 3.11 and a platform supported by the pinned PaddlePaddle dependencies. In the extracted project directory:

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app/main.py
```

Linux:
```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app/main.py
```

Dependency availability varies by platform. These instructions do not establish compatibility with every computer.

## Try it

1. Upload a clear JPG or PNG scoresheet photo. Recognition starts automatically.
2. Review the move table and any highlighted issues. Select a move to see its image crop and suggested alternatives.
3. Click a candidate or edit the table directly.
4. Compare the full game with the photo, confirm review, and download PGN.

If automatic table detection fails, the app offers manual table alignment. Strong perspective, curved paper, unusual layouts, and inconsistent handwriting remain difficult.

The `examples/` directory includes a sample scoresheet photograph and a reviewed PGN, plus review data. This is a demonstration, not an independent accuracy benchmark; some source-paper entries require correction.

## Implementation

Python · PaddleOCR/PaddlePaddle · OpenCV · python-chess · Streamlit

- `app/`: image processing, table reading, OCR, candidate ranking, validation and UI.
- `models/finetuned/`: bundled detection and recognition inference files.
- `data/dict_chess.txt`: recognition dictionary.
- `training/`: fine-tuning and evaluation materials.
- `tests/`: layout regression checks.
- [Model report](MODEL_REPORT.md): training details and evaluation limitations.
- [Detailed v5 notes in Chinese](docs/V5_NOTES_ZH.md): behavior, scope and known failures.

This version uses the v3 recognition weights with the v5 review interface. Reported cropped-text accuracy is not whole-page or complete-game accuracy. Existing training data may overlap data previously seen by the original model. Generalization to unseen writers and layouts needs stronger independent evaluation.

## Project background

The project began in 2024 to reduce the manual work of converting handwritten chess records into reusable digital games. Later development involved family collaboration and AI-assisted programming. This repository describes the current implementation; individual contributions and dates should be documented accurately in an application or project statement.

## Privacy

The application processes uploaded scoresheets locally. Do not publish private scoresheets or credentials when sharing a repository. The training image dataset and local virtual environments are not included in this package.
