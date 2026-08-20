from __future__ import annotations

import ctypes
from types import SimpleNamespace
from typing import Any

from rl_study import platform_metrics


class _FakeFunction:
    def __init__(self, callback: Any) -> None:
        self.callback = callback
        self.argtypes: list[object] = []
        self.restype: object | None = None

    def __call__(self, *args: object) -> object:
        return self.callback(*args)


def test_windows_memory_apis_use_typed_64_bit_values(
    monkeypatch: Any,
) -> None:
    def set_process_memory(_handle: object, counters: object, _size: object) -> int:
        counters._obj.peak_working_set_size = 987_654_321  # type: ignore[attr-defined]
        return 1

    def set_system_memory(status: object) -> int:
        status._obj.total_physical = 17_179_869_184  # type: ignore[attr-defined]
        return 1

    kernel32 = SimpleNamespace(
        GetCurrentProcess=_FakeFunction(lambda: ctypes.c_void_p(-1)),
        K32GetProcessMemoryInfo=_FakeFunction(set_process_memory),
        GlobalMemoryStatusEx=_FakeFunction(set_system_memory),
    )
    monkeypatch.setattr(
        platform_metrics,
        "_windows_loader",
        lambda: SimpleNamespace(kernel32=kernel32),
    )

    assert platform_metrics._windows_peak_memory_bytes() == 987_654_321
    assert platform_metrics._windows_system_memory_bytes() == 17_179_869_184


def test_platform_memory_metrics_are_positive() -> None:
    peak = platform_metrics.peak_memory_bytes()
    total = platform_metrics.system_memory_bytes()
    assert isinstance(peak, int) and peak > 0
    assert isinstance(total, int) and total > 0
