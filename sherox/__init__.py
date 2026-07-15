"""sherox — Speech Inference Toolkit"""

__version__ = "0.9.0"


class SherpaError(Exception):
    """Base exception for sherox errors.

    This exception is raised instead of calling sys.exit() in library code,
    allowing callers to handle errors gracefully (e.g., in tests or when
    used as a library).
    """


class ModelNotFoundError(SherpaError):
    """Raised when a required model file or directory is not found."""


class AudioError(SherpaError):
    """Raised for audio-related errors (file reading, device issues, etc.)."""


class ConfigError(SherpaError):
    """Raised for invalid configuration or arguments."""
