"""Content identities and atomic JSON publication."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(value):
    """Hash a canonical JSON-compatible value."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()


def file_digest(path):
    """Stream file hashes without loading videos or weights into memory."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    """Publish a complete file, even when the writer is interrupted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False
    ) as f:
        temporary = Path(f.name)
        try:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
