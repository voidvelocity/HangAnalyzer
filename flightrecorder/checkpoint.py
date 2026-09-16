"""ctypes wrapper for the optional CANN checkpoint adapter."""
from __future__ import annotations
import ctypes
import errno
import os


class CannCheckpointManager:
    def __init__(self, library: str, slots_per_stream: int = 4):
        self.lib = ctypes.CDLL(library)
        self.lib.flight_checkpoint_create_cann.argtypes = [ctypes.c_uint32]
        self.lib.flight_checkpoint_create_cann.restype = ctypes.c_void_p
        self.lib.flight_checkpoint_register_stream.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                                ctypes.c_void_p]
        self.lib.flight_checkpoint_register_stream.restype = ctypes.c_int
        self.lib.flight_checkpoint_submit.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                       ctypes.c_uint64, ctypes.c_uint64,
                                                       ctypes.POINTER(ctypes.c_uint64)]
        self.lib.flight_checkpoint_submit.restype = ctypes.c_int
        self.lib.flight_checkpoint_poll.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.lib.flight_checkpoint_poll.restype = ctypes.c_int
        self.lib.flight_checkpoint_start_poller.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                            ctypes.c_uint32]
        self.lib.flight_checkpoint_start_poller.restype = ctypes.c_int
        self.lib.flight_checkpoint_stop_poller.argtypes = [ctypes.c_void_p]
        self.lib.flight_checkpoint_destroy.argtypes = [ctypes.c_void_p]
        self.lib.flight_checkpoint_destroy.restype = ctypes.c_int
        self.handle = self.lib.flight_checkpoint_create_cann(slots_per_stream)
        if not self.handle:
            raise RuntimeError("unable to create CANN checkpoint manager")

    def register_stream(self, stream_id: int, stream_handle: int) -> None:
        rc = self.lib.flight_checkpoint_register_stream(self.handle, stream_id,
                                                        ctypes.c_void_p(stream_handle))
        if rc:
            raise OSError(rc, os.strerror(rc))

    def submit(self, stream_id: int, submitted_seq: int, checkpoint_id: int) -> int:
        generation = ctypes.c_uint64()
        rc = self.lib.flight_checkpoint_submit(self.handle, stream_id, submitted_seq,
                                               checkpoint_id, ctypes.byref(generation))
        if rc:
            raise OSError(rc, os.strerror(rc))
        return generation.value

    def poll(self, budget: int = 64) -> int:
        return self.lib.flight_checkpoint_poll(self.handle, budget)

    def start_poller(self, interval_us: int = 1000, budget: int = 64) -> None:
        rc = self.lib.flight_checkpoint_start_poller(self.handle, interval_us, budget)
        if rc:
            raise OSError(rc, os.strerror(rc))

    def stop_poller(self) -> None:
        if self.handle:
            self.lib.flight_checkpoint_stop_poller(self.handle)

    def close(self) -> None:
        if not self.handle:
            return
        self.stop_poller()
        rc = self.lib.flight_checkpoint_destroy(self.handle)
        if rc == errno.EBUSY:
            raise RuntimeError("checkpoint events still pending; keep manager alive until process exit")
        if rc:
            raise OSError(rc, os.strerror(rc))
        self.handle = None
