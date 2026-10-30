"""Module-level app instance for ASGI servers.

Uvicorn's reloader needs an import string rather than an object, so the default
application lives here. Tests build their own with ``create_app(...)``.
"""

from __future__ import annotations

from rbi_rag.api.app import create_app

app = create_app()
