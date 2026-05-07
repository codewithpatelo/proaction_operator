"""Detached watchdog for resilient experiment execution.

Monitors worker process and respawns on failure with backoff.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

PID_FILE = Path(".windsurf/checkpoints/watchdog.pid")
STATE_FILE = Path(".windsurf/checkpoints/watchdog_state.json")
LOG_FILE = Path(".windsurf/logs/watchdog.log")

POLL_INTERVAL = 15 * 60  # 15 minutes
MAX_CONSECUTIVE_FAILURES = 5
RATE_LIMIT_BACKOFF = 600  # 10 minutes


# ═══════════════════════════════════════════════════════════════════════════════
# State management
# ═══════════════════════════════════════════════════════════════════════════════

def load_state() -> dict[str, Any]:
    """Load watchdog state."""
    if not STATE_FILE.exists():
        return {
            "consecutive_failures": 0,
            "last_worker_pid": None,
            "last_rc": None,
            "started_at": None,
        }
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    """Save watchdog state atomically."""
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def log(message: str) -> None:
    """Log with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line)
    
    print(line, end="")


# ═══════════════════════════════════════════════════════════════════════════════
# Process management
# ═══════════════════════════════════════════════════════════════════════════════

def is_process_alive(pid: int) -> bool:
    """Check if process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def write_pid(pid: int) -> None:
    """Write watchdog PID to file."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(pid))


def read_pid() -> int | None:
    """Read watchdog PID from file."""
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except (ValueError, IOError):
        return None


def cleanup_pid() -> None:
    """Remove PID file."""
    if PID_FILE.exists():
        PID_FILE.unlink()


def spawn_worker(args: list[str]) -> subprocess.Popen:
    """Spawn worker process detached."""
    log(f"Spawning worker: {' '.join(args)}")
    
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    
    return subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main watchdog loop
# ═══════════════════════════════════════════════════════════════════════════════

def watchdog_loop(
    worker_args: list[str],
    max_runtime_hours: float = 5.0,
) -> int:
    """Main watchdog loop.
    
    Args:
        worker_args: Command to spawn worker
        max_runtime_hours: Maximum total runtime
    
    Returns:
        Final exit code
    """
    start_time = time.time()
    max_runtime = max_runtime_hours * 3600
    
    state = load_state()
    state["started_at"] = time.time()
    save_state(state)
    
    worker = None
    worker_pid = None
    
    log("Watchdog starting...")
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_runtime:
            log(f"Max runtime ({max_runtime_hours}h) exceeded, aborting")
            return 86
        
        if worker is None or not is_process_alive(worker_pid):
            if worker is not None:
                state = load_state()
                rc = state.get("last_rc")
                log(f"Worker exited with rc={rc}")
                
                if rc == 0:
                    log("Worker completed successfully")
                    return 0
                elif rc == 75:
                    log(f"Rate limit hit, backing off for {RATE_LIMIT_BACKOFF}s")
                    state["consecutive_failures"] += 1
                    save_state(state)
                    time.sleep(RATE_LIMIT_BACKOFF)
                elif rc == 86:
                    log("Critical error (rc=86), halting")
                    return 86
                else:
                    state["consecutive_failures"] += 1
                    save_state(state)
                    if state["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
                        log(f"Max consecutive failures ({MAX_CONSECUTIVE_FAILURES}) reached")
                        return 86
            
            # Check provider budget exhaustion
            flag_path = Path("provider_budget_exhausted.flag")
            if flag_path.exists():
                provider = flag_path.read_text().split()[0]
                log(f"Provider {provider} budget exhausted")
            
            worker = spawn_worker(worker_args)
            worker_pid = worker.pid
            state["last_worker_pid"] = worker_pid
            save_state(state)
            log(f"Spawned worker PID {worker_pid}")
        
        time.sleep(POLL_INTERVAL)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Watchdog for experiment runner")
    parser.add_argument("--worker-cmd", nargs="+", default=["python", "-m", "exp.runner", "--benchmark"],
                        help="Command to run worker")
    parser.add_argument("--max-hours", type=float, default=5.0,
                        help="Maximum runtime in hours")
    parser.add_argument("--stop", action="store_true",
                        help="Stop running watchdog")
    
    args = parser.parse_args()
    
    if args.stop:
        pid = read_pid()
        if pid and is_process_alive(pid):
            os.kill(pid, 15)  # SIGTERM
            log(f"Sent stop signal to watchdog PID {pid}")
        else:
            log("No watchdog running")
        return 0
    
    # Check if already running
    existing_pid = read_pid()
    if existing_pid and is_process_alive(existing_pid):
        log(f"Watchdog already running (PID {existing_pid})")
        return 1
    
    # Write PID
    write_pid(os.getpid())
    
    try:
        rc = watchdog_loop(args.worker_cmd, args.max_hours)
    finally:
        cleanup_pid()
    
    return rc


if __name__ == "__main__":
    sys.exit(main())
