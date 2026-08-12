"""
Monkey patch for importlib_metadata to handle None metadata case.
Must be imported before ANY other imports (including langchain).
"""
import importlib.metadata
import importlib_metadata

_original_metadata_version = importlib_metadata.version

def patched_version(distribution_name):
    try:
        return _original_metadata_version(distribution_name)
    except Exception as e:
        if "'NoneType'" in str(e) or "NoneType" in str(e):
            return "0.0.0"
        raise

importlib.metadata.version = patched_version
importlib_metadata.version = patched_version

# Also patch the inner metadata getter so md_none doesn't crash
_original_metadata = importlib_metadata.Distribution.metadata.fget if hasattr(importlib_metadata.Distribution.metadata, 'fget') else importlib_metadata.Distribution.metadata

importlib_metadata._meta.Distribution.metadata = property(
    lambda self: self._read_metadata() if hasattr(self, '_read_metadata') else {},
)