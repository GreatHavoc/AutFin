"""
Pytest configuration with monkey patches for importlib.metadata.version.
This must be loaded before any other imports to prevent MetadataNotFound errors.
"""
import importlib.metadata
import importlib_metadata

# Monkey patch for importlib.metadata.version to handle missing metadata.
# This is a workaround for Python 3.14 + importlib_metadata + huggingface_hub compatibility issue.
def _safe_version(dist_name):
    try:
        return importlib.metadata.version(dist_name)
    except Exception:
        return "0.0.0"

importlib_metadata.version = _safe_version
importlib.metadata.version = _safe_version