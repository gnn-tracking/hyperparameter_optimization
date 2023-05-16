from __future__ import annotations

from typing import Iterable


def pop(dct: dict, keys: Iterable[str]) -> dict:
    """Return a new dict with the given keys popped from the given dict."""
    return {k: dct.pop(k) for k in keys}
