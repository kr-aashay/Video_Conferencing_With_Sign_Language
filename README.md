# Aashay's Sign Lang — Sign Language Video Conferencing

A real-time sign language video conferencing system where a signer performs hand gestures on camera and captions are automatically generated and displayed for all participants in the call — like live subtitles, but for sign language.

---

## Demo Flow

```
Signer performs gesture → MediaPipe detects it → Gemini generates caption → Caption appears for everyone in the call
```

---

## Features

- 🤟 **Real-time gesture recognition** — MediaPipe detects 7 ASL/ISL gestures directly in the browser
- 💬 **AI-powered captions** — Gemini 2.0 Flash converts sign glosses into natural English sentences
- 📹 **Video conferencing** — ZegoCloud handles peer-to-peer video/audio between participants
- 🗄️ **Caption history** — Every caption is saved to MongoDB so late joiners can catch up
- ⚡ **Low latency** — ~1–3s end-to-end from gesture to caption
- 🔒 **Auth** — Register/login with bcrypt-hashed passwords and session tokens

---

## Supported Gestures

| Gesture | Label |
|---------|-------|
| Open Palm | HELLO |
| Thumb Up | YES |
| Thumb Down | NO |
| Pointing Up | I |
| Closed Fist | STOP |
| Victory ✌️ | PEACE |
| ILoveYou 🤟 | LOVE |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI (Python) |
| Database | MongoDB (Motor async driver) |
| Video Conferencing | ZegoCloud UIKit |
| Gesture Detection | MediaPipe Tasks Vision (browser, WASM) |
| Sign Language Model | Custom Bi-LSTM + CTC (PyTorch) |
| Caption Generation | Gemini 2.0 Flash |
| Authentication | bcrypt + itsdangerous |
| Real-time | WebSockets |
| Frontend | Vanilla JS + Jinja2 |

---

## Models

### MediaPipe Gesture Recognizer
Pre-trained Google model running in the browser via WebAssembly. Processes camera frames at 30fps and detects hand gestures with a majority-vote confirmation over 10 frames (~0.3s hold time).

### Custom Bi-LSTM (ISL-CSLTR)
A custom-trained Continuous Sign Language Recognition model trained on 11,980 Indian Sign Language videos.

```
Input (T, 225)  ← pose + both hands landmarks
    ↓
SpatialEmbedding  Conv1D(225→256) + BatchNorm + GELU
    ↓
TemporalEncoder   2× Bi-LSTM (512 hidden units)
    ↓
AttentionPool     learned weighted average over frames
    ↓
ClassifierHead    Linear(512→256→vocab_size)
```

- **Accuracy:** 91.2%
- **Dataset:** ISL-CSLTR (11,980 videos, 641 classes)

### Gemini 2.0 Flash
Converts sign glosses (e.g. `HELLO I LOVE`) into natural English sentences (e.g. `"Hello, I love you."`). Uses per-room conversation history for pronoun resolution and context.

---

## Project Structure

```
├── main.py                  # FastAPI server — auth, pages, WebSocket caption bridge
├── api/                     # Modular inference bridge (Bi-LSTM + Ollama/OpenAI)
├── cslr_model/              # Custom sign language model (PyTorch)
├── templates/               # Jinja2 HTML pages
├── static/                  # CSS + MediaPipe WASM bundle
├── frontend/                # Advanced sender/receiver UI
├── tests/                   # Unit + integration tests
├── train.py                 # Model training script
├── retrain.py               # Incremental retraining
├── validate.py              # Model validation
├── record_signs.py          # Record new sign samples
├── gesture_recognizer.task  # MediaPipe model file (served locally)
├── requirements.txt
└── .env.example             # Environment variable template
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/kr-aashay/Video_Conferencing_With_Sign_Language.git
cd Video_Conferencing_With_Sign_Language
```

### 2. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=aashay_signlang
SECRET_KEY=your-secret-key

ZEGO_APP_ID=your_zego_app_id
ZEGO_SERVER_SECRET=your_zego_server_secret

GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash
```

- **ZegoCloud keys** → [console.zegocloud.com](https://console.zegocloud.com) (free)
- **Gemini key** → [aistudio.google.com](https://aistudio.google.com/app/apikey) (free)
- **MongoDB** → local install or [MongoDB Atlas](https://www.mongodb.com/atlas) (free tier)

### 5. Run
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**

---

## Usage

1. Register and log in
2. Click **Create Meeting** — share the room link with others
3. Click **Enable Sign Mode**
4. Hold a hand gesture for ~0.3 seconds
5. Caption appears at the bottom of the screen for **all participants**

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MONGODB_URI` | MongoDB connection string |
| `MONGODB_DB` | Database name |
| `SECRET_KEY` | Session signing secret |
| `ZEGO_APP_ID` | ZegoCloud App ID |
| `ZEGO_SERVER_SECRET` | ZegoCloud Server Secret |
| `GEMINI_API_KEY` | Google Gemini API key |
| `GEMINI_MODEL` | Gemini model name (default: `gemini-2.0-flash`) |
| `DEVICE` | Inference device: `cpu`, `cuda`, or `mps` |

---

## License

MIT
