"""Simple in-process TTL cache. Keys should include source and parameters."""
from __future__ import annotations
import time
from threading import RLock
from typing import Any

_CACHE: dict[str, tuple[float, Any]] = {}
_LOCK = RLock()

def cache_get(key: str, ttl_seconds: float = 300) -> Any | None:
    with _LOCK:
        item = _CACHE.get(key)
        if item is None:
            return None
        created, value = item
        if time.time() - created > ttl_seconds:
            _CACHE.pop(key, None)
            return None
        return value

def cache_set(key: str, value: Any) -> None:
    with _LOCK:
        _CACHE[key] = (time.time(), value)

def cache_clear() -> None:
    with _LOCK:
        _CACHE.clear()
