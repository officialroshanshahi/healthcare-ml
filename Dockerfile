FROM python:3.14-slim

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy project specification files and source code
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/

# install dependencies
RUN pip install --no-cache-dir .

# expose ports for FastAPI and Gradio
EXPOSE 8000 7860

# run uvicorn initialization and gradio side-by-side
CMD ["sh", "-c", "python -m api.uvicorn_init & python -m app.gradio"]