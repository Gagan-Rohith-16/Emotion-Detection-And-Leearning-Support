# Emotion Detection and Learning Support

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-2ea44f?style=for-the-badge)](https://emotion-detection-and-leearning-support.onrender.com)

Emotion Detection and Learning Support is a Streamlit-based AI platform that helps students describe learning difficulties, detects likely emotions, and provides personalized study guidance.

Live MVP: https://emotion-detection-and-leearning-support.onrender.com

The app combines:
- BiLSTM classifier
- BERT classifier
- Keyword-based rule engine
- Fusion logic for final emotion prediction
- Gemini-powered coaching guidance (with safe fallback when API is unavailable)

## Features

- User authentication with private browser-cookie persistence
- Emotion prediction from free-text learning challenges
- Multi-model fusion output with confidence and agreement score
- Personalized guidance from Gemini (`gemini-2.5-flash`) with robust fallback mode
- Dashboard and analytics views for historical emotion trends
- SQLite-backed storage for users and prediction history

## Supported Emotions

- Bored
- Confident
- Confused
- Curious
- Frustrated

## Project Structure

```
Emotion_Detection_Learning_Support/
|- app.py
|- requirements.txt
|- database/
|  |- database.py
|  |- models.py
|  |- schema.sql
|- prediction/
|  |- emotion_pipeline.py
|  |- emotion_fusion.py
|  |- keyword_rules.py
|- models/
|  |- bilstm.py
|  |- bert.py
|  |- artifacts/
|- gemini/
|  |- guidance.py
|- ui/
|  |- auth.py
|  |- navigation.py
|  |- pages.py
|- training/
|  |- train_bilstm.py
|  |- train_bert.py
|- dataset/
	|- raw/
	|- processed/
	|- scripts/
```

## Tech Stack

- Python 3.12+
- Streamlit
- TensorFlow / Keras (BiLSTM)
- PyTorch + Transformers (BERT)
- Pandas / NumPy / scikit-learn
- SQLite
- Google Gen AI SDK (`google-genai`)

## Setup

1. Clone the repository

```bash
git clone https://github.com/Gagan-Rohith-16/Emotion-Detection-And-Leearning-Support.git
cd Emotion-Detection-And-Leearning-Support
```

2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Configure environment variables (optional but recommended)

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

If no key is provided, the app still works and returns deterministic fallback guidance.

## Run the App

```bash
streamlit run app.py
```

The app initializes the SQLite database automatically using `database/schema.sql`.

## Deploy on Render

This repository is now configured for Docker-based deployment on Render.

Deployment files:

- `Dockerfile`
- `render.yaml`
- `.streamlit/config.toml`
- `.dockerignore`

### Steps

1. Push your latest code to GitHub.
2. In Render, create a new **Blueprint** and select this repository.
3. Render reads `render.yaml` and creates the web service automatically.
4. Set `GEMINI_API_KEY` in Render Environment Variables (optional).
5. Deploy.

### Notes for production

- The app listens on Render's injected `PORT` automatically.
- If `GEMINI_API_KEY` is missing, guidance falls back to deterministic offline responses.
- If BERT artifacts are not available in deployment, the prediction pipeline still works using BiLSTM + Keyword Rules + Fusion.
- For persistent user history, attach a Render Disk and store `database/emotion_support.db` on that mounted path.

## Model and Data Notes

- `venv/` is excluded from version control.
- Large local model file `models/artifacts/bert/model.safetensors` is excluded from Git to avoid GitHub's file-size limit.
- Included artifacts currently allow running the app without retraining.

## Training

Train BiLSTM:

```bash
python training/train_bilstm.py
```

Train BERT:

```bash
python training/train_bert.py
```

## Dataset Preparation

Data scripts are available in `dataset/scripts/`:

- `prepare_dataset.py`
- `generate_bored_dataset.py`
- `merge_dataset.py`
- `split_dataset.py`

## Testing

Quick pipeline test:

```bash
python test_pipeline.py
```

Other test files:

- `test_bert.py`
- `test_gemini_key.py`
- `test_google_api.py`
- `training/test_bilstm.py`

## Security and Privacy

- Passwords are hashed with scrypt and never stored in plaintext.
- SQL queries are parameterized in `database/database.py`.
- Emotion predictions are assistance signals, not medical diagnoses.

## Future Improvements

- Add CI workflow for linting, tests, and packaging checks
- Add model artifact versioning with Git LFS or external storage
- Add API layer for mobile/web client integration

## Author

Gagan Rohith