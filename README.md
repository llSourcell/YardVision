# YardVision

Edge-deployable computer vision application scaffold, optimized for Python 3.11+, Poetry, and strict quality tooling. Provides a Typer CLI, a simple processing pipeline, analytics placeholders, Docker setup, and CI workflow.

## Features

- Typer-powered CLI with `run` command
- Simple CV pipeline with OpenCV (dummy inference + annotation)
 - Video processing pipeline with YOLOv8 detection + ByteTrack MOT using `supervision`
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

# Process a video file and save annotated output
poetry run yardvision process-video ./input.mp4 ./output.mp4 --model yolov8m.pt
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

## Configuration

Basic settings are defined in `vision/core/config.py`. Environment variables from `.env` are loaded automatically if present. You can add parsing of TOML/YAML files in `load_config()` when needed.

## CI/CD

GitHub Actions workflow runs lint, type-check, and tests on pushes and PRs to `main`.

## Development Tips

- Use `loguru` for structured logging and `rich` for better tracebacks and CLI output.
- Extend `FrameProcessor._dummy_infer` with a real ONNX model using `onnxruntime`.
- Add analytics publishing in `vision/analytics/` and wire into the pipeline.

## License

MIT

