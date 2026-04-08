FROM python:3.10-slim

WORKDIR /app

# Install git (needed for openenv-core from github)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Try installing openenv-core (non-fatal if it fails)
RUN pip install --no-cache-dir "openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv.git@v0.2.3" || true

# Copy the rest of the application
COPY . .

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]