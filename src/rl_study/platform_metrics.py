"""Cross-platform process metrics using only the Python standard library."""

from __future__ import annotations

import ctypes
import sys


def _windows_peak_memory_bytes() -> int | None:
    """Return PeakWorkingSetSize through the Windows Process Status API."""

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("page_fault_count", ctypes.c_ulong),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
        ]

    loader = getattr(ctypes, "windll", None)
    if loader is None:
        return None
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    process = loader.kernel32.GetCurrentProcess()
    succeeded = loader.psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    )
    if not succeeded:
        return None
    return int(counters.peak_working_set_size)


def peak_memory_bytes() -> int | None:
    """Return peak resident process memory on Windows, macOS, or Linux."""

    if sys.platform == "win32":
        return _windows_peak_memory_bytes()

    import resource

    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if sys.platform == "darwin" else maximum * 1024)
