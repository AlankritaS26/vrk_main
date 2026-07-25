"""
VRK Kiosk — one-command launcher (self-healing).

    venv\\Scripts\\python.exe run.py

Starts, in one terminal with prefixed logs:
  [BACKEND]   uvicorn backend.main:app     (port 8001)
  [DETECT]    backend.detection            (only after backend is healthy)
  [FRONTEND]  npm start                    (./frontend, port 3000)

Self-healing:
  * If ports 8001/3000 are held by a STALE python/node process from a
    previous run, it is killed automatically — no manual taskkill.
  * If a port is held by some unrelated program, we refuse (safely) and
    tell you which program it is.

Ctrl+C stops all three services cleanly.
"""

import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable                       # the venv python that launched us
BACKEND_URL = "http://127.0.0.1:8001"

# process names we are allowed to auto-kill when they squat on our ports
KILLABLE = {"python.exe", "pythonw.exe", "uvicorn.exe", "node.exe"}

COLORS = {"BACKEND": "\033[96m", "DETECT": "\033[93m", "FRONTEND": "\033[92m", "RUN": "\033[95m"}
RESET = "\033[0m"

procs: list = []


def say(name: str, text: str):
    print(f"{COLORS.get(name, '')}[{name}]{RESET} {text}")


# ─────────────────────────────── port self-healing ──────────────────────────

def pids_on_port(port: int) -> set[int]:
    """Parse netstat for PIDs LISTENING on the port (Windows + Unix fallback)."""
    pids = set()
    try:
        if os.name == "nt":
            out = subprocess.run(["netstat", "-ano"], capture_output=True,
                                 text=True, timeout=10).stdout
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and f":{port}" in parts[1] and "LISTENING" in line:
                    if parts[-1].isdigit():
                        pids.add(int(parts[-1]))
        else:
            out = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True,
                                 text=True, timeout=10).stdout
            pids = {int(p) for p in out.split() if p.strip().isdigit()}
    except Exception:
        pass
    return pids


def process_name(pid: int) -> str:
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, timeout=10).stdout
            if '","' in out:
                return out.split('","')[0].strip('"').lower()
        else:
            out = subprocess.run(["ps", "-p", str(pid), "-o", "comm="],
                                 capture_output=True, text=True, timeout=10).stdout
            return out.strip().lower()
    except Exception:
        pass
    return "unknown"


def kill_pid(pid: int):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True, timeout=10)
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def free_port(port: int, label: str) -> bool:
    """Kill stale dev processes on the port. True if the port is free after."""
    pids = pids_on_port(port)
    if not pids:
        return True
    for pid in pids:
        name = process_name(pid)
        if name in KILLABLE:
            say("RUN", f"Port {port} held by stale {name} (PID {pid}) — killing it.")
            kill_pid(pid)
        else:
            say("RUN", f"Port {port} is used by '{name}' (PID {pid}) — not a kiosk "
                       f"process, refusing to kill it automatically.")
            say("RUN", f"Close that program, or change the {label} port, then rerun.")
            return False
    # give the OS a moment to release the socket
    for _ in range(10):
        if not pids_on_port(port):
            return True
        time.sleep(0.5)
    say("RUN", f"Port {port} still busy after cleanup — try once more in a few seconds.")
    return False


# ─────────────────────────────── service plumbing ───────────────────────────

def _pipe(name: str, proc: subprocess.Popen):
    for line in iter(proc.stdout.readline, b""):
        try:
            print(f"{COLORS.get(name, '')}[{name}]{RESET} "
                  f"{line.decode('utf-8', errors='replace').rstrip()}")
        except Exception:
            continue
    proc.stdout.close()


def start(name: str, args, cwd=None, shell=False, env=None) -> subprocess.Popen:
    proc = subprocess.Popen(
        args, cwd=cwd or ROOT, shell=shell, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    procs.append(proc)
    threading.Thread(target=_pipe, args=(name, proc), daemon=True).start()
    return proc


def wait_for_backend(timeout: float = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BACKEND_URL + "/health", timeout=2):
                return True
        except Exception:
            time.sleep(1.5)
    return False


def shutdown(*_):
    say("RUN", "Stopping all services...")
    for p in procs:
        try:
            if os.name == "nt":
                p.send_signal(signal.CTRL_BREAK_EVENT)
            p.terminate()
        except Exception:
            pass
    time.sleep(2)
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass
    # sweep anything that survived (npm spawns child node processes)
    for port in (8001, 3000):
        for pid in pids_on_port(port):
            if process_name(pid) in KILLABLE:
                kill_pid(pid)
    sys.exit(0)


# ─────────────────────────────────── main ───────────────────────────────────

def main():
    signal.signal(signal.SIGINT, shutdown)
    os.system("")                          # enable ANSI colors on Windows

    say("RUN", "Checking ports...")
    if not free_port(8001, "backend"):
        sys.exit(1)
    if not free_port(3000, "frontend"):
        sys.exit(1)

    say("RUN", "Starting backend...")
    start("BACKEND", [PY, "-m", "uvicorn", "backend.main:app",
                      "--host", "127.0.0.1", "--port", "8001"])

    say("RUN", "Waiting for backend health (models loading — first boot takes longer)...")
    if not wait_for_backend():
        say("RUN", "Backend never became healthy — check [BACKEND] logs above.")
        shutdown()

    say("RUN", "Backend healthy. Starting camera detection...")
    start("DETECT", [PY, "-m", "backend.detection"])

    say("RUN", "Starting frontend (npm start)...")
    env = os.environ.copy()
    env["BROWSER"] = "none"          # we open the kiosk browser ourselves, with flags
    start("FRONTEND", "npm start", cwd=os.path.join(ROOT, "frontend"),
          shell=True, env=env)

    # Open the kiosk browser with autoplay ENABLED so the greeting can speak
    # before any user gesture — this is the standard kiosk deployment flag.
    def open_kiosk_browser():
        time.sleep(10)                 # let the CRA dev server come up
        flags = "--autoplay-policy=no-user-gesture-required --app=http://localhost:3000"
        for browser in ("chrome", "msedge"):
            try:
                subprocess.Popen(f'start "" {browser} {flags}', shell=True)
                say("RUN", f"Opened kiosk window in {browser} (autoplay enabled).")
                return
            except Exception:
                continue
        say("RUN", "Could not auto-open a browser — open http://localhost:3000 manually.")
    threading.Thread(target=open_kiosk_browser, daemon=True).start()

    say("RUN", "All services launching. Kiosk UI: http://localhost:3000  |  Ctrl+C stops everything.")
    try:
        while True:
            time.sleep(1)
            for p in list(procs):
                if p.poll() is not None and p.returncode not in (0, None):
                    say("RUN", f"A service exited (code {p.returncode}). "
                               "Others keep running; Ctrl+C to stop all.")
                    procs.remove(p)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()