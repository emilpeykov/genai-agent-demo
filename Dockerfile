# Runs this project with ZERO local prerequisites beyond Docker itself -- no
# Python install, no Ollama, no API key. Builds + runs the Gradio UI by
# default (the "Rule-based agent" tab works immediately; the Ollama-dependent
# tabs show a friendly in-UI message instead of crashing -- see
# app/gradio_app.py). Override CMD to run the FastAPI service instead (see
# the two commented examples at the bottom of this file / the README's
# "Quickstart" section).
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer across rebuilds
# that only change application code (not requirements.txt).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Leave OPENAI_API_KEY unset / LLM_BACKEND=mock so the container runs fully
# offline out of the box -- see .env.example for every other optional knob.
ENV LLM_BACKEND=mock \
    PYTHONUNBUFFERED=1

EXPOSE 7860 8000

# Default: the Gradio UI (python app.py binds 0.0.0.0 so it's reachable from
# outside the container -- see demo.launch() call below).
CMD ["python", "app.py"]

# To run the FastAPI service instead:
#   docker build -t genai-agent-demo .
#   docker run -p 8000:8000 genai-agent-demo uvicorn app.main:app --host 0.0.0.0 --port 8000
#
# To point either at a REAL Ollama running on the host machine instead of the
# mock provider:
#   docker run -p 7860:7860 -e LLM_BACKEND=ollama \
#     -e OLLAMA_BASE_URL=http://host.docker.internal:11434 genai-agent-demo

