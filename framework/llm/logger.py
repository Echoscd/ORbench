"""
logger.py - LLM interaction logging.

Logs every LLM call (prompt + response) to structured log files under logs/.
Each generation batch gets its own log file with timestamp.

Log format: one JSON object per line (JSONL), easy to parse and grep.
"""

import os
import json
import time
import threading
from typing import Optional

from ..task import ORBENCH_ROOT


LOGS_DIR = os.path.join(ORBENCH_ROOT, "logs")

# Module-level lock for thread-safe writes
_write_lock = threading.Lock()

# Current log file path (set by init_logger)
_log_file: Optional[str] = None


def init_logger(tag: str = "llm") -> str:
    """
    Initialize a new log file for this session.

    Args:
        tag: A short label (e.g. model name or "batch")

    Returns:
        Path to the log file.
    """
    global _log_file
    os.makedirs(LOGS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    _log_file = os.path.join(LOGS_DIR, f"{tag}_{ts}.jsonl")
    return _log_file


def get_log_file() -> Optional[str]:
    """Return the current log file path, or None if not initialized."""
    return _log_file


def log_request(
    model_id: str,
    model_string: str,
    provider: str,
    prompt: str,
    task_id: str = "",
    level: int = 0,
    sample_id: int = 0,
    max_tokens: int = 0,
    temperature: float = 0.0,
):
    """Log an outgoing LLM request (before the call)."""
    _write_entry({
        "event": "request",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "model_string": model_string,
        "provider": provider,
        "task_id": task_id,
        "level": level,
        "sample_id": sample_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "prompt_chars": len(prompt),
        "prompt": prompt,
    })


def log_response(
    model_id: str,
    task_id: str = "",
    level: int = 0,
    sample_id: int = 0,
    content: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    latency_ms: float = 0.0,
):
    """Log a successful LLM response."""
    _write_entry({
        "event": "response",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "task_id": task_id,
        "level": level,
        "sample_id": sample_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
        "response_chars": len(content),
        "response": content,
    })


def log_error(
    model_id: str,
    task_id: str = "",
    level: int = 0,
    sample_id: int = 0,
    error: str = "",
    error_type: str = "",
    attempt: int = 0,
):
    """Log a failed LLM call."""
    _write_entry({
        "event": "error",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "task_id": task_id,
        "level": level,
        "sample_id": sample_id,
        "error_type": error_type,
        "error": error,
        "attempt": attempt,
    })


def _write_entry(entry: dict):
    """Append a JSON line to the current log file (thread-safe)."""
    global _log_file
    if _log_file is None:
        init_logger()

    with _write_lock:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

