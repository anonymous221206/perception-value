"""mmengine.fileio's local-path behaviour: bytes in, bytes out."""
from pathlib import Path


def get(path):
    return Path(path).read_bytes()


def get_text(path, encoding="utf-8"):
    return Path(path).read_text(encoding=encoding)


def put(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(obj if isinstance(obj, (bytes, bytearray)) else str(obj).encode())
