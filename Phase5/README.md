# Neural Network Music Visualizer

Emotion-aware real-time music visualizer powered by a CNN-LSTM model.

## Architecture

Audio → Phase1 (features) → Phase2 (CNN-LSTM) → Phase3 (mapping) → Phase4 (renderer)

## Run locally without Docker

    pip install -r phase1/requirements.txt
    pip install -r phase2/requirements.txt
    pip install -r phase3/requirements.txt
    pip install -r phase4/requirements.txt

    cd phase4
    python main.py --mode web

Open phase4/server/index.html in your browser.

## Run with Docker

    cd NNMusicVisualizer
    docker-compose -f phase5/docker-compose.yml up --build

Open phase4/server/index.html in your browser.

## Run all tests

    cd phase1 && pytest tests/ -v
    cd phase2 && pytest tests/ -v
    cd phase3 && pytest tests/ -v
    cd phase4 && pytest tests/ -v
    cd phase5 && pytest integration/ -v

## Cloud deployment

Backend → Railway
Frontend → Vercel