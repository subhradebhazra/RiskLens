"""Bounded HTTP retries for transient provider faults only."""
from __future__ import annotations
import time
import requests

_TRANSIENT_STATUS={408,425,429,500,502,503,504}

def request_with_retry(method: str, url: str, *, timeout: float=8.0, attempts: int=3, **kwargs) -> requests.Response:
    """Retry timeouts/connections and selected transient HTTP statuses with backoff.

    Authentication, permission, malformed-request and other 4xx responses are not
    retried. Caller-facing code should sanitize exceptions before display.
    """
    if not 1<=attempts<=4: raise ValueError("attempts must be between 1 and 4")
    last_exc=None
    for attempt in range(attempts):
        try:
            response=requests.request(method.upper(),url,timeout=timeout,**kwargs)
            if response.status_code in _TRANSIENT_STATUS and attempt<attempts-1:
                retry_after=response.headers.get("Retry-After")
                try: delay=min(max(float(retry_after),0),4) if retry_after is not None else min(0.5*(2**attempt),4)
                except ValueError: delay=min(0.5*(2**attempt),4)
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response
        except (requests.Timeout,requests.ConnectionError) as exc:
            last_exc=exc
            if attempt>=attempts-1: raise
            time.sleep(min(0.5*(2**attempt),4))
    if last_exc is not None: raise last_exc
    raise RuntimeError("HTTP request did not complete.")
