FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
COPY config/ ./config/

RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 1000 agent
USER agent

WORKDIR /workspace

ENTRYPOINT ["agent"]
CMD ["--help"]