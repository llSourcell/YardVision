# syntax=docker/dockerfile:1.7
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    CMAKE_BUILD_PARALLEL_LEVEL=2

# System deps for CV: build tools, libjpeg, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Poetry
RUN pip install "poetry==${POETRY_VERSION}"

# Copy project metadata first for better caching
COPY pyproject.toml README.md ./

# Install dependencies only
RUN poetry install --no-root --only main

# Copy the rest of the source
COPY src ./src

# Install project in editable mode for CLI access
RUN poetry install

ENTRYPOINT ["python", "-m", "yardvision"]

