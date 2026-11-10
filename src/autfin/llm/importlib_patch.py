"""
Monkey patch for importlib.metadata to handle the None metadata case.

Import-safe on any Python: the `importlib_metadata` backport is only present in
older environments, so fall back to the stdlib module (3.12+ has the API this
patch needs) instead of failing the import of the whole package.
"""
import importlib.metadata

try:  # backport, when installed
    import importlib_metadata as _backend
except ImportError:  # pragma: no cover - Python 3.12+
    _backend = importlib.metadata

_original_metadata_version = _backend.version


def patched_version(distribution_name):
    try:
        return _original_metadata_version(distribution_name)
    except Exception as e:
        if "NoneType" in str(e):
            return "0.0.0"
        raise


_backend.version = patched_version
importlib.metadata.version = patched_version

# Only the backport exposes `_meta.Distribution`; stdlib has a `_meta` module but not that class.
if getattr(getattr(_backend, "_meta", None), "Distribution", None) is not None:  # pragma: no cover - backport only
    _backend._meta.Distribution.metadata = property(
        lambda self: self._read_metadata() if hasattr(self, "_read_metadata") else {},
    )
