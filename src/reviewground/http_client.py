from __future__ import annotations

import os
import ssl
import urllib.request
from functools import lru_cache
from typing import Optional


def _resolve_certifi_cafile() -> Optional[str]:
    try:
        import certifi

        return certifi.where()
    except Exception:
        pass
    try:
        from pip._vendor import certifi as pip_certifi

        return pip_certifi.where()
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_ssl_context() -> Optional[ssl.SSLContext]:
    """Build an HTTPS context with a stable CA bundle fallback."""
    env_cafile = (os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE") or "").strip()
    if env_cafile:
        try:
            return ssl.create_default_context(cafile=env_cafile)
        except Exception:
            pass

    certifi_cafile = _resolve_certifi_cafile()
    if certifi_cafile:
        try:
            return ssl.create_default_context(cafile=certifi_cafile)
        except Exception:
            pass

    try:
        return ssl.create_default_context()
    except Exception:
        return None


def urlopen_with_ssl(req: urllib.request.Request, timeout: float):
    context = get_ssl_context()
    if context is None:
        return urllib.request.urlopen(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout, context=context)
