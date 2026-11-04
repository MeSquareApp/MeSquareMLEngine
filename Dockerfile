# MeSquareMLEngine API with embedded R and AHAprevent for PREVENT risk scores.
FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        r-base \
        r-base-dev \
        libgomp1 \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY PREVENT/AHAprevent_1.1.0.tar.gz /tmp/AHAprevent_1.1.0.tar.gz
RUN R CMD INSTALL /tmp/AHAprevent_1.1.0.tar.gz \
    && rm /tmp/AHAprevent_1.1.0.tar.gz

COPY app.py train_model.py upload_data.py evaluate_lab_stage.py ./
COPY config ./config
COPY src ./src
COPY sql ./sql
COPY supabase ./supabase

EXPOSE 8000

CMD ["uv", "run", "fastapi", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
