FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN apt-get update \
    && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

COPY src ./src
COPY jobs ./jobs
COPY data ./data
COPY metrics ./metrics

COPY .dvc ./.dvc
COPY dvc.yaml .
COPY dvc.lock .
COPY params.yaml .
COPY .gitignore .

# Create Git repository for DVC
RUN git init \
    && git config user.email "docker@aero-metrics.local" \
    && git config user.name "aero-metrics" \
    && git add -A \
    && git commit -m "Initial repository state"

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]