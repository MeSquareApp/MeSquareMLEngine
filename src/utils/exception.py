class MeSquareError(Exception):
    """Base error for MeSquareMLEngine."""


class IngestError(MeSquareError):
    """Raised when a dataset cannot be loaded, transformed, or verified."""
