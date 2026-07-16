# MergenVision backend — Phase 1 Sprint 01
# Build context must be the repository root so that backend/ is copied in.

FROM python:3.12-slim-bookworm AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency metadata and install only runtime packages, using the lockfile
# as a constraints file so versions remain reproducible.
COPY backend/pyproject.toml backend/README.md ./
COPY backend/requirements.lock ./

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -c requirements.lock -e .

# Copy application source.
COPY backend/app ./app

# ------------------------------------------------------------------------------
# Runtime image
# ------------------------------------------------------------------------------

FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r mergenvision \
    && useradd -r -g mergenvision -d /app -s /sbin/nologin mergenvision

WORKDIR /app

COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

USER mergenvision

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["python", "-c", "print('MergenVision backend container is ready.')"]
