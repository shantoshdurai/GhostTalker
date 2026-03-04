# GhostTalker - Deployment Guide

This guide covers deploying GhostTalker in different environments.

## Prerequisites

- Docker & Docker Compose
- Python 3.10+
- GPU (optional, but recommended for faster TTS)

## Quick Start (Docker)

### 1. Clone the repository

```bash
git clone https://github.com/shantoshdurai/GhostTalker.git
cd GhostTalker
```

### 2. Configure environment variables

Copy and edit the `.env` file:

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

### 3. Build and run with Docker Compose

```bash
# Build containers
docker compose build

# Start all services
docker compose up -d

# View logs
docker compose logs -f
```

### 4. Access the application

Once running, open your browser at: `http://localhost:5000`

## Local Development (without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

## Project Structure

```
GhostTalker/
├── .agent/workflows/  # Agent workflow definitions
├── .github/           # GitHub Actions workflows
├── clone/             # Voice cloning utilities
├── docker/            # Docker configuration files
├── images/            # Static images
├── models/            # AI model files
├── static/            # Web static assets
├── templates/         # HTML templates
├── tts/               # Text-to-Speech engine
├── tts_cache/         # Cached TTS audio
├── utils/             # Utility functions
├── app.py             # Main application
└── docs/              # Documentation
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MODEL_PATH` | Path to AI model files | Yes |
| `TTS_ENGINE` | TTS engine to use | Yes |
| `PORT` | Application port (default: 5000) | No |
| `DEBUG` | Enable debug mode | No |

## Troubleshooting

- **TTS not working?** Check model files exist in `models/` directory
- **Docker build fails?** Ensure Docker daemon is running
- **Voice cloning errors?** Verify audio samples are in the correct format

---

*Last updated: March 2026*
