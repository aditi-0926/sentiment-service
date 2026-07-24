# Sentiment Service

A real-time sentiment classification API powered by a fine-tuned DistilBERT model, containerized with Docker, and deployed through a CI/CD pipeline with an automated model-quality gate.

![CI](https://github.com/aditi-0926/sentiment-service/actions/workflows/ci.yml/badge.svg)
![CD](https://github.com/aditi-0926/sentiment-service/actions/workflows/cd.yml/badge.svg)

## Overview

This project treats model performance as a first-class CI check, not just an afterthought. Every push runs automated tests **and** an evaluation gate that blocks deployment if the model's accuracy or F1 score falls below a defined threshold — the same way a broken unit test would block a bad code change.

**Pipeline:** `Fine-tune DistilBERT → Serve via FastAPI → Test → Quality Gate → Docker Build → Push to GHCR`

## Architecture

```
                ┌─────────────────┐
                │  tweet_eval      │
                │  (sentiment)     │
                └────────┬─────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Fine-tune       │
                │  DistilBERT      │
                │  (training/)     │
                └────────┬─────────┘
                         │
              writes metrics.json
                         │
                         ▼
                ┌─────────────────┐
                │  FastAPI app     │
                │  (app/)          │
                │  /predict        │
                │  /health         │
                └────────┬─────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │         GitHub Actions          │
        │                                  │
        │  CI  → lint, API tests,          │
        │        quality gate (min         │
        │        accuracy / F1)            │
        │                                  │
        │  CD  → docker build, push to     │
        │        GHCR (on merge to main)   │
        └────────────────────────────────┘
                         │
                         ▼
        ghcr.io/aditi-0926/sentiment-service
```

## Model

| | |
|---|---|
| Base model | `distilbert-base-uncased` |
| Task | 3-class sentiment classification (negative / neutral / positive) |
| Dataset | [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (sentiment config) |
| Framework | Hugging Face `transformers` + `Trainer` |

### Current metrics

| Metric | Score |
|---|---|
| Accuracy | 0.660 |
| F1 (macro) | 0.654 |

These are checked automatically against `MIN_ACCURACY = 0.60` and `MIN_F1 = 0.55` in [`tests/test_eval_gate.py`](tests/test_eval_gate.py) — the build fails if a retrained model regresses below these thresholds.

> Note: this model was trained on a subset of the training data for fast iteration. A full-dataset run (see `training/train.py`) typically pushes accuracy into the 72–76% range.

## Project structure

```
sentiment-service/
├── .github/workflows/
│   ├── ci.yml              # lint, tests, quality gate
│   └── cd.yml               # docker build + push to GHCR
├── app/
│   ├── main.py               # FastAPI app (/health, /predict)
│   ├── model.py               # model loading + inference
│   └── schemas.py             # request/response models
├── training/
│   └── train.py                # fine-tuning + evaluation script
├── tests/
│   ├── test_api.py              # API endpoint tests
│   └── test_eval_gate.py         # model quality gate
├── models/final/                  # fine-tuned model + tokenizer (Git LFS)
├── metrics.json                    # latest evaluation metrics
├── Dockerfile
└── requirements.txt
```

## Running locally

### 1. Set up the environment
```bash
conda create -n sentiment python=3.11 -y
conda activate sentiment
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Train the model (optional — a trained model is already included via Git LFS)
```bash
python training/train.py
```
This writes `models/final/` (weights + tokenizer) and `metrics.json`.

### 3. Run the API
```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Test it
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"I absolutely love this!\"}"
```
```json
{"label": "positive", "confidence": 0.9886}
```

## Running with Docker

### Build locally
```bash
docker build -t sentiment-service .
docker run -p 8000:8000 sentiment-service
```

### Or pull the published image
```bash
docker pull ghcr.io/aditi-0926/sentiment-service:latest
docker run -p 8000:8000 ghcr.io/aditi-0926/sentiment-service:latest
```

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check, returns `{"status": "ok"}` |
| `/predict` | POST | Returns predicted sentiment label + confidence |

**Request:**
```json
{ "text": "This is the worst experience ever." }
```

**Response:**
```json
{ "label": "negative", "confidence": 0.9529 }
```

## CI/CD pipeline

**CI** (`.github/workflows/ci.yml`) runs on every push and pull request to `main`:
1. Checks out code (with Git LFS)
2. Installs dependencies
3. Lints with `ruff`
4. Runs API tests
5. Runs the **quality gate** — fails the build if `metrics.json` accuracy/F1 fall below threshold

**CD** (`.github/workflows/cd.yml`) runs on every push to `main`:
1. Builds the Docker image
2. Pushes it to GitHub Container Registry (`ghcr.io/aditi-0926/sentiment-service`)

`main` is protected — pull requests can't merge unless the CI check passes, making the quality gate an enforced part of the workflow rather than an advisory one.

## Tech stack

- **Model:** Hugging Face `transformers`, `datasets`, PyTorch (CPU)
- **API:** FastAPI, Uvicorn, Pydantic
- **Testing:** pytest
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Registry:** GitHub Container Registry (GHCR)
- **Model storage:** Git LFS

## Future improvements

- [ ] Full-dataset training run (GPU) for a stronger baseline model
- [ ] MLflow or Weights & Biases for experiment tracking across training runs
- [ ] Live deployment (Render / Fly.io) with a public demo endpoint
- [ ] Load testing (`locust` / `k6`) for the `/predict` endpoint
- [ ] Docker image vulnerability scanning (Trivy) in CI
