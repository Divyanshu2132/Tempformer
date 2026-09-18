"""
task_lock.py -- cross-process / cross-job task claiming via atomic lock files.

Multiple independent SLURM jobs (e.g. one per GPU) scan the same directory
tree on a shared filesystem and must not process the same input file twice.
Claiming is done with os.open(..., O_CREAT | O_EXCL), which is an atomic
"create-if-not-exists" at the filesystem level on Lustre/POSIX -- exactly one
caller across all processes/hosts can win the race for a given lock file.
"""
import os
import socket
import time
from pathlib import Path

STALE_SECONDS = 2 * 60 * 60  # locks older than this are assumed abandoned (killed/OOM job)


def _lock_path(target: Path) -> Path:
    return target.parent / (target.name + ".lock")


def try_claim(target: Path, stale_seconds: int = STALE_SECONDS) -> bool:
    """Attempt to claim `target` for processing.

    Returns True if this call won the claim (caller must produce `target`
    and then call release(target)). Returns False if `target` already
    exists or another worker currently holds a live lock on it.
    """
    if target.exists():
        return False

    lock = _lock_path(target)
    for attempt in range(2):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if attempt == 0 and _is_stale(lock, stale_seconds):
                _clear_stale(lock)
                continue
            return False
        else:
            os.write(fd, f"{socket.gethostname()}:{os.getpid()}:{time.time()}\n".encode())
            os.close(fd)
            return True
    return False


def release(target: Path) -> None:
    """Release a lock previously won with try_claim."""
    try:
        _lock_path(target).unlink()
    except FileNotFoundError:
        pass


def _is_stale(lock: Path, stale_seconds: int) -> bool:
    try:
        return (time.time() - lock.stat().st_mtime) > stale_seconds
    except FileNotFoundError:
        return False


def _clear_stale(lock: Path) -> None:
    try:
        lock.unlink()
    except FileNotFoundError:
        pass
