"""Start the bundled local-only server if needed, without a visible helper window."""
import os
import subprocess
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent


def ensure_server():
    session = requests.Session()
    session.trust_env = False
    try:
        if session.get("http://127.0.0.1:11434/api/tags", timeout=2).ok:
            return
    except requests.RequestException:
        pass
    executable = ROOT / ".local-ai/ollama/ollama.exe"
    if not executable.exists():
        raise SystemExit("Local AI missing: run setup_local_ai.ps1 first.")
    env = dict(os.environ, OLLAMA_MODELS=str(ROOT / ".local-ai/models"), OLLAMA_NO_CLOUD="1",
               OLLAMA_HOST="127.0.0.1:11434", OLLAMA_CONTEXT_LENGTH="8192", OLLAMA_MAX_LOADED_MODELS="1")
    with (ROOT / ".local-ai/server.log").open("ab") as log:
        subprocess.Popen([str(executable), "serve"], cwd=ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
    for _ in range(15):
        try:
            if session.get("http://127.0.0.1:11434/api/tags", timeout=1).ok:
                return
        except requests.RequestException:
            time.sleep(1)
    raise SystemExit("Local AI did not start. Check .local-ai/server.log.")


if __name__ == "__main__":
    ensure_server()
