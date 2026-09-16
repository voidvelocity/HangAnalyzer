"""Small explicit instrumentation API for vLLM/vLLM-Ascend integration."""
from __future__ import annotations
import ctypes
import os
from pathlib import Path
from .format import TYPES


class Recorder:
    def __init__(self, directory: str, rank: int, device: int, library: str, capacity: int = 1_048_576):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"rank{rank}-pid{os.getpid()}.flight"
        self.lib = ctypes.CDLL(library)
        self.lib.flight_init.argtypes = [ctypes.c_char_p, ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint64]
        self.lib.flight_init.restype = ctypes.c_int
        self.lib.flight_record.argtypes = [ctypes.c_uint16, ctypes.c_uint32, ctypes.c_uint64,
                                           ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint16]
        self.lib.flight_record.restype = ctypes.c_uint64
        self.lib.flight_close.argtypes = []
        err = self.lib.flight_init(os.fsencode(self.path), rank, device, capacity)
        if err:
            raise OSError(err, os.strerror(err), str(self.path))

    def record(self, name: str, *, stream: int = 0, correlation: int = 0,
               arg0: int = 0, arg1: int = 0, flags: int = 0) -> int:
        return self.lib.flight_record(TYPES[name], stream, correlation, arg0, arg1, flags)

    def close(self) -> None:
        self.lib.flight_close()
