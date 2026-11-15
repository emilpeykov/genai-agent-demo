"""Entrypoint for the Hugging Face Spaces **Gradio SDK**.

Spaces auto-detects this root-level ``app.py`` and runs it directly (the SDK
expects either a module-level ``demo`` Gradio object, or a script that calls
``.launch()`` -- this does both, so it also works unchanged for a plain
``python app.py`` locally).

See ``app/gradio_app.py`` for the actual UI (kept there, not duplicated here,
so ``python -m app.gradio_app`` -- the way this project has always been run
locally/in development -- keeps working exactly as before).

Deploying to a Space: see the "Deploying to Hugging Face Spaces" section in
README.md for the exact steps (SDK: gradio, this file as the entrypoint).
Note that the **Rule-based agent** tab is the one guaranteed to work on a
hosted Space with no further setup (fully offline mock LLM provider, no API
key) -- the **ReAct agent** tab requires a real Ollama server reachable at
``OLLAMA_BASE_URL``, which a default Space won't have.
"""
import os

from app.gradio_app import build_app

demo = build_app()

if __name__ == "__main__":
    # server_name="0.0.0.0" makes this reachable from outside a Docker
    # container (default "127.0.0.1" would only be reachable from inside
    # it); harmless for plain local use too, still reachable at
    # http://localhost:7860. Port is configurable via $PORT for platforms
    # (Spaces, Docker, etc.) that assign one.
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))


