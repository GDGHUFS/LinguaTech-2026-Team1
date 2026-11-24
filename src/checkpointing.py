"""Atomic model/result persistence: interrupted writes never replace a good file."""
import os
from pathlib import Path
import tempfile
import joblib
import torch


def atomic_dump(path, value, serializer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    os.close(fd)
    try:
        serializer(value, temporary)
        with open(temporary, 'r+b') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def save_torch(path, value):
    atomic_dump(path, value, torch.save)


def save_joblib(path, value):
    atomic_dump(path, value, joblib.dump)
