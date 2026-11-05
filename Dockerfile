# MeSquareMLEngine API with embedded R and AHAprevent for PREVENT risk scores.
# rpy2>=3.6 needs R>=4.5 (R_getVar). Bookworm's apt R is older, so use CRAN.
FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg \
        dirmngr \
    && gpg --batch --keyserver keyserver.ubuntu.com \
        --recv-key 95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7 \
    && gpg --batch --armor --export 95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7 \
        > /etc/apt/trusted.gpg.d/cran_debian_key.asc \
    && echo "deb http://cloud.r-project.org/bin/linux/debian bookworm-cran46/" \
        > /etc/apt/sources.list.d/cran-r.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        r-base \
        r-base-dev \
        libgomp1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && R --version | head -1

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
