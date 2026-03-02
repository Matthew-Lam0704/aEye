# aEye

aEye is a Flask-based, real-time computer vision assistant focused on environmental awareness.  
It detects people and objects, estimates relative risk/distance, infers basic human actions, and reads scene text through OCR.

## Features

- Real-time detection pipeline powered by YOLO models (objects + pose).
- Person action inference (`standing`, `waving`, `sitting`, `fallen`) with temporal smoothing.
- Hazard-first logic with direction and distance tiers (`left/center/right`, `near/close/very close`).
- OCR endpoint with PaddleOCR primary + Tesseract fallback.
- Annotated frame streaming for frontend display.
- Mobile-friendly browser flow (`/` landing, `/detect` live detection view).

## Tech Stack

- Python + Flask
- OpenCV
- Ultralytics YOLO
- PaddleOCR + PaddlePaddle
- Tesseract OCR (fallback)

## Project Structure

- `app.py` - main Flask app, inference pipeline, OCR, and API routes
- `templates/` - frontend pages (`landing.html`, `detect.html`)
- `static/app.js` - browser camera capture + API integration
- `requirements.txt` - Python dependencies

## Setup

### 1) Prerequisites

- Python 3.10+ (recommended)
- `pip`
- Tesseract installed locally

For Apple Silicon/Homebrew:

```bash
brew install tesseract
```

> Note: `app.py` currently points Tesseract to `/opt/homebrew/bin/tesseract`.
> If your install path is different, update:
>
> `pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"`

### 2) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Run the app

```bash
python3 app.py
```

Server starts on:

- `http://127.0.0.1:5050`
- `http://0.0.0.0:5050` (LAN access)

## Usage

1. Open `http://127.0.0.1:5050`
2. Go to the detection page.
3. Allow camera permissions.
4. The frontend sends frames to backend endpoints for:
   - scene/object/person analysis
   - OCR text extraction

## API Endpoints

- `GET /` - landing page
- `GET /detect` - detection page
- `GET /detections` - latest processed state JSON
- `POST /infer` - process a frame and return detections + annotated image
- `POST /ocr` - OCR on a frame (PaddleOCR first, Tesseract fallback)

## Notes on OCR

- PaddleOCR is used when available and initialized correctly.
- If PaddleOCR cannot initialize, the app automatically falls back to Tesseract.
- First PaddleOCR run may download model files, so internet access may be required.

## Optional: Share with phone outside local network

```bash
ngrok http 5050
```

Then open the generated `https://...` URL on your phone.

## Troubleshooting

- **`PaddleOCR unavailable`**
  - Ensure both packages are installed in your active venv:
    - `paddleocr`
    - `paddlepaddle`
- **No OCR text returned**
  - Verify camera frames are reaching backend.
  - Confirm Tesseract path is valid on your machine.
- **Model host/network warnings**
  - PaddleOCR may show host connectivity warnings during startup. This is expected if model hosts are blocked; local/fallback OCR may still work.
