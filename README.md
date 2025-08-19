# YardVision

Privacy-first, edge-deployable computer vision application optimized for Python 3.11+, Poetry, and strict quality tooling. It features YOLOv8 detection, ByteTrack MOT, spatial analytics (heatmaps, zones, dwell-time, near-miss), and GDPR-compliant selective blurring. [Live Demo](https://yardvision-demo-df5c32-caa3aefc0f47.herokuapp.com/)

![Demo](demo.gif)

## Why YardVision

- **Privacy-first**: On-device, GDPR-conscious selective blurring of people.
- **Edge-ready**: CPU-friendly baseline with clean path to GPU acceleration.
- **Operational analytics**: Zones, dwell counts, and near-miss heuristics for safety.
- **Production hygiene**: CI, typing, linting, pre-commit, containers.

## Features

- Typer-powered CLI with `run` command
- YOLOv8 detection + ByteTrack MOT using `supervision`
- Spatial analytics: heatmap trails, polygon zones with dwell counts, near-miss highlighting
- GDPR selective blurring: blur only the `person` class before overlays
- Strict typing (mypy --strict), linting (ruff), tests (pytest)
- Pre-commit hooks configured
- Dockerfile optimized for CPU-based CV workloads
- docker-compose for local development with webcam passthrough
- GitHub Actions CI (lint, type-check, test)

## Project Structure

```
src/
  yardvision/
    __init__.py
    __main__.py
    cli.py
  vision/
    __init__.py
    core/
      config.py
    pipeline/
      processor.py
    analytics/
      __init__.py
      events.py

.github/workflows/ci.yml
Dockerfile
docker-compose.yml
pyproject.toml
.pre-commit-config.yaml
.gitignore
README.md
```

## Getting Started

### Prerequisites

- Python 3.11+
- Poetry 1.8+

### Installation

```bash
poetry install
poetry run pre-commit install
```

### Run

```bash
# From webcam 0, headless (no display)
poetry run yardvision run --source 0

# From video file with display window
poetry run yardvision run --source ./sample.mp4 --display

# Process a video file with analytics + privacy blurring and save annotated output
poetry run yardvision process-video ./input.mp4 ./output.mp4 --model yolov8n.pt
```

### Tests

```bash
poetry run pytest -q
```

### Lint and Type Check

```bash
poetry run ruff check . && poetry run ruff format --check
poetry run mypy --strict --python-version 3.11
```

## Docker

Build and run the container:

```bash
docker build -t yardvision:latest .
docker run --rm -it --device /dev/video0:/dev/video0 yardvision:latest run --source 0
```

Or with docker-compose:

```bash
docker compose up --build
```

Notes:
- `opencv-python-headless` is used for headless environments. For local GUI display inside the container, additional X11 setup is required; otherwise use host Python for display.
- To leverage GPUs later, switch base image to CUDA-enabled and add the appropriate runtime (e.g., `--gpus all`).

## Architecture Overview

```
src/
  yardvision/
    cli.py                  # Typer CLI: run, process-video
  vision/
    core/config.py          # AppConfig, settings
    pipeline/processor.py   # FrameProcessor (live), VideoProcessor (batch)
    analytics/              # (extensible) analytics events
```

- **Detection**: Ultralytics `YOLO` models (e.g., `yolov8n.pt`).
- **Tracking**: `supervision.ByteTrack` for MOT IDs.
- **Analytics**: `HeatMapAnnotator`, `PolygonZone` with `PolygonZoneAnnotator`, dwell counting.
- **Privacy**: `BlurAnnotator` over `person` detections before drawing other overlays.
- **Output**: `VideoSink` writes MP4 (`mp4v` codec).

### Key Pipeline Steps

1) Run YOLO on each frame and convert to `supervision.Detections`.
2) Filter to `person`, `car`, `truck` classes.
3) Update ByteTrack for persistent IDs.
4) Blur only `person` detections on the base frame.
5) Overlay heatmap, zone polygon and dwell count.
6) Compute near-miss (pixel-distance heuristic) and highlight in red.
7) Save annotated frame.

## Configuration

Basic settings are defined in `vision/core/config.py`. Environment variables from `.env` are loaded automatically if present. You can add parsing of TOML/YAML files in `load_config()` when needed.

## CI/CD

GitHub Actions workflow runs lint, type-check, and tests on pushes and PRs to `main`.

## Development Tips

- Use `loguru` for structured logging and `rich` for better tracebacks and CLI output.
- Replace heuristic near-miss with calibrated physical distance using homography.
- Add RTSP ingestion and sink for live deployments.
- Promote zones to config and expose via CLI.

## License

MIT

