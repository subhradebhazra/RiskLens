"""Small security helpers; credentials are not persisted by the app."""
from __future__ import annotations
import re
from collections.abc import Iterable

_SECRET_PATTERNS = [re.compile(r"(?i)(api[_ -]?key|token|secret|authorization)(\s*[:=]\s*)([^\s,;]+)")]

def mask_secret(value: str | None) -> str:
    if not value:
        return "Not configured"
    if len(value) <= 8:
        return "•" * len(value)
    return f"{value[:3]}{'•' * min(12, len(value)-6)}{value[-3:]}"

def sanitize_text(message: object, secrets: Iterable[str] = ()) -> str:
    """Remove known secrets and common key/token assignments from diagnostics."""
    result = str(message)
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(r"\1\2[REDACTED]", result)
    return result[:2000]

def validate_http_url(url: str) -> bool:
    return url.startswith("https://") or url.startswith("http://")
