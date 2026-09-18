"""Single-window orchestrator for DeepInterview.

Runs Edge-TTS, Agent API, Live Voice Worker, and Web UI as managed child processes
in ONE single console window. Shuts all services down cleanly when closed or on Ctrl+C.
Zero performance loss — each service still runs on its own process.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / ".deepinterview" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_EXE = REPO_ROOT / "apps" / "agent" / ".venv" / "Scripts" / "python.exe"
if not PYTHON_EXE.exists():
    PYTHON_EXE = Path(sys.executable)

processes: list[tuple[str, subprocess.Popen]] = []


def sync_env() -> dict[str, str]:
    """Sync root .env to apps/agent/.env and apps/web/.env.local, and parse for child processes."""
    env_file = REPO_ROOT / ".env"
    parsed: dict[str, str] = {}
    if env_file.exists():
        content = env_file.read_text(encoding="utf-8")
        (REPO_ROOT / "apps" / "agent" / ".env").write_text(content, encoding="utf-8")
        (REPO_ROOT / "apps" / "web" / ".env.local").write_text(content, encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                parsed[k.strip()] = v.strip()
    return parsed


def stop_all() -> None:
    print("\nShutting down all DeepInterview services...")
    for name, proc in processes:
        try:
            # On Windows, taskkill /T kills process tree (including node children)
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    print("All services stopped cleanly.")


def main() -> None:
    print("=" * 60)
    print("           DeepInterview AI Mock Interview Platform")
    print("=" * 60)
    print()

    env_vars = sync_env()
    proc_env = {**os.environ, **env_vars}

    # 1. Edge TTS Bridge
    edge_log = open(LOGS_DIR / "edge_tts.log", "w", encoding="utf-8")
    print("[1/4] Starting Microsoft Edge Neural Voice Bridge (Port 8880)...")
    p_edge = subprocess.Popen(
        [str(PYTHON_EXE), "services/edge_tts_server.py"],
        cwd=str(REPO_ROOT),
        stdout=edge_log,
        stderr=subprocess.STDOUT,
        env=proc_env,
    )
    processes.append(("Edge-TTS", p_edge))

    # 2. Agent API
    agent_log = open(LOGS_DIR / "agent_api.log", "w", encoding="utf-8")
    print("[2/4] Starting Agent API Backend (Port 8000)...")
    p_api = subprocess.Popen(
        [str(PYTHON_EXE), "-m", "uvicorn", "deepinterview_agent.app:app", "--port", "8000"],
        cwd=str(REPO_ROOT / "apps" / "agent"),
        stdout=agent_log,
        stderr=subprocess.STDOUT,
        env=proc_env,
    )
    processes.append(("Agent-API", p_api))

    # 3. Live Voice Worker
    worker_log = open(LOGS_DIR / "worker.log", "w", encoding="utf-8")
    print("[3/4] Starting Live Voice Worker (LiveKit Agent)...")
    p_worker = subprocess.Popen(
        [str(PYTHON_EXE), "-m", "deepinterview_agent.worker", "start"],
        cwd=str(REPO_ROOT / "apps" / "agent"),
        stdout=worker_log,
        stderr=subprocess.STDOUT,
        env=proc_env,
    )
    processes.append(("Worker", p_worker))

    # 4. Next.js Web App
    web_log = open(LOGS_DIR / "web.log", "w", encoding="utf-8")
    print("[4/4] Starting Next.js Web Application (Port 3000)...")
    p_web = subprocess.Popen(
        ["pnpm.cmd", "--filter", "@deepinterview/web", "dev"],
        cwd=str(REPO_ROOT),
        stdout=web_log,
        stderr=subprocess.STDOUT,
        env=proc_env,
        shell=True,
    )
    processes.append(("Web", p_web))

    print()
    print("All services launched successfully in the background!")
    print(f"Service logs are saved to: {LOGS_DIR}")
    print()
    print("Waiting 4 seconds before opening browser...")
    time.sleep(4)

    print("Opening http://localhost:3000 ...")
    webbrowser.open("http://localhost:3000")

    print()
    print("=" * 60)
    print("DeepInterview is LIVE!")
    print("-> Keep this SINGLE terminal window open while practicing.")
    print("-> Press Ctrl+C in this window to stop all services.")
    print("=" * 60)
    print()

    try:
        while True:
            # Check if any critical process died unexpectedly
            for name, proc in processes:
                code = proc.poll()
                if code is not None:
                    print(f"\n[WARNING] {name} exited unexpectedly with code {code}.")
                    print(f"Check the log file: {LOGS_DIR / (name.lower() + '.log')}")
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_all()


if __name__ == "__main__":
    main()
