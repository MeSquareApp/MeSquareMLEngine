"""Small helpers shared by the ingest path."""

import re

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_safe_identifier(name: str) -> bool:
    """Return whether a name is safe to use as a SQL identifier."""
    return _IDENTIFIER.fullmatch(name) is not None
