"""Cross-platform process metrics using only the Python standard library."""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Protocol, cast


class _CFunction(Protocol):
    argtypes: object
    restype: object

    def __call__(self, *args: object) -> object: ...


class _Kernel32(Protocol):
    GetCurrentProcess: _CFunction
    K32GetProcessMemoryInfo: _CFunction
    GlobalMemoryStatusEx: _CFunction


class _Psapi(Protocol):
    GetProcessMemoryInfo: _CFunction


class _WindowsLoader(Protocol):
    kernel32: _Kernel32
    psapi: _Psapi


def _windows_loader() -> _WindowsLoader | None:
    return cast("_WindowsLoader | None", getattr(ctypes, "windll", None))


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

    loader = _windows_loader()
    if loader is None:
        return None
    kernel32 = loader.kernel32
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p
    try:
        get_process_memory_info = kernel32.K32GetProcessMemoryInfo
    except AttributeError:
        get_process_memory_info = loader.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    get_process_memory_info.restype = ctypes.c_int
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    process = get_current_process()
    succeeded = get_process_memory_info(process, ctypes.byref(counters), counters.cb)
    if not succeeded:
        return None
    return int(counters.peak_working_set_size)


def _windows_system_memory_bytes() -> int | None:
    """Return total physical RAM through GlobalMemoryStatusEx."""

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    loader = _windows_loader()
    if loader is None:
        return None
    global_memory_status = loader.kernel32.GlobalMemoryStatusEx
    global_memory_status.argtypes = [ctypes.POINTER(MemoryStatus)]
    global_memory_status.restype = ctypes.c_int
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not global_memory_status(ctypes.byref(status)):
        return None
    return int(status.total_physical)


def peak_memory_bytes() -> int | None:
    """Return peak resident process memory on Windows, macOS, or Linux."""

    if sys.platform == "win32":
        return _windows_peak_memory_bytes()

    import resource

    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if sys.platform == "darwin" else maximum * 1024)


def system_memory_bytes() -> int | None:
    """Return total physical memory on Windows, macOS, or Linux."""

    if sys.platform == "win32":
        return _windows_system_memory_bytes()
    for page_size_name in ("SC_PAGE_SIZE", "SC_PAGESIZE"):
        try:
            page_size = os.sysconf(page_size_name)
            physical_pages = os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, OSError, ValueError):
            continue
        if isinstance(page_size, int) and isinstance(physical_pages, int):
            return page_size * physical_pages
    return None
