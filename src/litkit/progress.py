# litkit/progress.py
"""
Progress reporting and output utilities for litkit.

This module centralizes all progress display, phase banners, and output
utilities that were previously scattered throughout cli.py.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import TextIO

# Re-export the progress lock and helpers from embeddings.base for consistency
from litkit.embeddings.base import (
    _PROGRESS_LOCK,
    progress_is_append as _progress_is_append,
    progress_newline as _progress_newline,
    progress_write as _progress_write,
)

__all__ = [
    "eprint",
    "Progress",
    "Pulse",
    "phase",
    "QUIET",
    "set_quiet",
    "is_quiet",
    "print_parallelism_summary",
]


# -------- Global quiet mode --------
# Set LITKIT_QUIET=1 to squelch startup banners that print before args are parsed.
import os


def is_quiet() -> bool:
    """Check if quiet mode is enabled (reads env var on each call).
    
    This function checks os.environ each time to ensure changes to
    LITKIT_QUIET (e.g., from --quiet flag) take effect immediately.
    """
    return os.environ.get("LITKIT_QUIET", "0") == "1"


# Legacy alias for backward compatibility (deprecated - use is_quiet())
# NOTE: This is frozen at import time. For runtime checks, use is_quiet().
QUIET = is_quiet()


def set_quiet(value: bool) -> None:
    """Set quiet mode via environment variable.
    
    This sets LITKIT_QUIET in os.environ so all modules see the change.
    """
    global QUIET
    if value:
        os.environ["LITKIT_QUIET"] = "1"
    else:
        os.environ.pop("LITKIT_QUIET", None)
    # Update frozen QUIET for legacy code that uses it
    QUIET = value


def eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush, respecting quiet mode for empty messages.
    
    This is the standard way to emit operator-facing messages in litkit.
    Checks env var on each call so changes to LITKIT_QUIET take effect immediately.
    """
    if is_quiet() and not msg:
        return
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


def phase(name: str, stream: TextIO | None = None) -> None:
    """Print a simple phase banner (operator log only).
    
    Ensures the previous progress line is terminated before printing.
    """
    stream = stream or sys.stderr
    try:
        _progress_newline(stream)
    except Exception:
        pass
    stream.write(f"[phase] {name}\n")
    stream.flush()
    try:
        with _PROGRESS_LOCK:
            pass  # Just ensure lock is released
    except Exception:
        pass


class Progress:
    """Lightweight progress line reporter (stderr), dependency-free.
    
    Displays a single updating line showing count, percentage, and rate.
    
    Usage:
        prog = Progress("Processing items", total=100)
        for item in items:
            process(item)
            prog.tick()
        prog.finish()
    
    Args:
        force_append: If True, always emit newlines between updates instead
                      of using TTY-aware single-line overwriting. Useful for
                      long-running operations where log visibility matters
                      (e.g., IVF-PQ training).
    """
    
    def __init__(
        self,
        label: str,
        total: int | None = None,
        start: int = 0,
        min_interval: float = 0.2,
        stream: TextIO | None = None,
        emit_final_line: bool = True,
        force_append: bool = False,
    ):
        self.label = label
        self.total = total if (total is not None and total > 0) else None
        self.done = int(start)
        self.start_ts = time.time()
        self.last_ts = 0.0
        self.min_interval = float(min_interval)
        self.stream = stream if stream is not None else sys.stderr
        self.emit_final_line = bool(emit_final_line)
        self.force_append = bool(force_append)
        self._last_len = 0

    def _write_line(self, s: str) -> None:
        if self.force_append:
            # Always emit newlines (explicit append mode)
            with _PROGRESS_LOCK:
                self.stream.write(s + "\n")
                self.stream.flush()
        else:
            _progress_write(s, self.stream)

    def _fmt(self) -> str:
        elapsed = max(1e-3, time.time() - self.start_ts)
        rate = self.done / elapsed
        if self.total is None:
            return f"[progress] {self.label}: {self.done}  ({rate:.1f}/s)"
        pct = 100.0 * self.done / max(1, self.total)
        return (
            f"[progress] {self.label}: {self.done}/{self.total}"
            f"  ({pct:.1f}%)  {rate:.1f}/s"
        )

    def tick(self, inc: int = 1, force: bool = False) -> None:
        """Increment counter and optionally refresh display."""
        self.done += inc
        now = time.time()
        if force or (now - self.last_ts) >= self.min_interval:
            self._write_line(self._fmt())
            self.last_ts = now

    def finish(self, emit_final_line: bool | None = None) -> None:
        """Finalize progress display.
        
        Args:
            emit_final_line: Override instance default for emitting final line
        """
        do_emit = (
            self.emit_final_line if emit_final_line is None 
            else bool(emit_final_line)
        )
        if do_emit:
            _progress_write(self._fmt(), self.stream)
        _progress_newline(self.stream)


class Pulse:
    """Background heartbeat that refreshes a single progress line with elapsed time.
    
    Useful for long-running operations where you want to show the user that
    something is happening, even if there's no meaningful progress count.
    
    Usage:
        pulse = Pulse("Training model")
        try:
            train_model()  # Long operation
        finally:
            pulse.stop()
    """

    def __init__(
        self, 
        label: str, 
        period: float = 0.5, 
        stream: TextIO | None = None
    ):
        self.label = label
        self.period = float(period)
        self.stream = stream if stream is not None else sys.stderr
        self._stop = threading.Event()
        self._t0 = time.time()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def _print_line(self, s: str) -> None:
        _progress_write(s, self.stream)

    def _run(self) -> None:
        while not self._stop.is_set():
            elapsed = int(time.time() - self._t0)
            self._print_line(
                f"[progress] {self.label}: training... {elapsed}s elapsed"
            )
            self._stop.wait(self.period)

    def stop(self) -> None:
        """Stop the pulse and emit a final status line."""
        self._stop.set()
        try:
            self._thr.join(timeout=2.0)
        except Exception:
            pass
        
        elapsed = int(time.time() - self._t0)
        # Terminate the line so the next writer doesn't append mid-line.
        if _progress_is_append():
            _progress_write(
                f"[progress] {self.label}: completed in {elapsed}s",
                self.stream,
            )
            _progress_newline(self.stream)
        else:
            self._print_line(
                f"[progress] {self.label}: training… {elapsed}s elapsed"
            )
            _progress_newline(self.stream)


def print_parallelism_summary(
    requested_gpu_devices: list[str] | None = None,
    actual_gpu_stats: dict[str, int] | None = None,
    requested_xml_workers: int | None = None,
    actual_xml_stats: dict[int, int] | None = None,
) -> None:
    """Print parallelism utilization summary to stderr.
    
    This summary shows requested vs actual resource utilization for
    debugging multi-GPU and multi-CPU configurations.
    
    Args:
        requested_gpu_devices: List of GPU devices requested (e.g., ["cuda:0", "cuda:1"])
        actual_gpu_stats: Dict of device -> batch_count from EmbeddingPool._devices_used
        requested_xml_workers: Number of XML parse workers requested
        actual_xml_stats: Dict of thread_id -> file_count from get_xml_thread_stats()
    """
    lines = ["", "=== PARALLELISM SUMMARY ==="]
    
    # GPU summary
    if requested_gpu_devices is not None:
        req_count = len(requested_gpu_devices)
        if actual_gpu_stats:
            actual_count = len(actual_gpu_stats)
            details = ", ".join(
                f"{dev}: {cnt} batches" 
                for dev, cnt in sorted(actual_gpu_stats.items())
            )
            status = ""
            if actual_count < req_count:
                status = " ⚠️ UNDERUTILIZED"
            lines.append(
                f"[gpu] requested={req_count} devices, "
                f"actual={actual_count} devices used [{details}]{status}"
            )
        else:
            lines.append(f"[gpu] requested={req_count} devices, actual=0 (no batches processed)")
    
    # XML summary
    if requested_xml_workers is not None:
        if actual_xml_stats:
            actual_count = len(actual_xml_stats)
            # Shorten thread IDs for readability
            details = ", ".join(
                f"tid{tid % 10000}: {cnt} files"
                for tid, cnt in sorted(actual_xml_stats.items())
            )
            status = ""
            if actual_count < requested_xml_workers:
                status = " ⚠️ UNDERUTILIZED"
            lines.append(
                f"[xml] requested={requested_xml_workers} workers, "
                f"actual={actual_count} threads used [{details}]{status}"
            )
        else:
            lines.append(f"[xml] requested={requested_xml_workers} workers, actual=0 (no files parsed)")
    
    # Only print if we have something to report
    if len(lines) > 2:
        for line in lines:
            eprint(line)


# Convenience aliases matching cli.py names (for easier migration)
_eprint = eprint
_Progress = Progress
_Pulse = Pulse
_phase = phase
