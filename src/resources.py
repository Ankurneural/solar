"""Keep the laptop usable while experiments run.

The machine (16 CPU threads, 8 GB GPU, 63 GB RAM) is also the user's daily
workstation and hung twice when jobs saturated it. Every entry point calls
`limit_resources()` first:
  * process priority BELOW_NORMAL, so interactive apps always win the CPU
  * thread pools (torch, OpenMP/MKL, OpenCV) capped at ~60% of the CPU threads
  * GPU memory capped at `gpu_mem_frac` of the card
and GPU loops call `GpuThrottle.step()` once per iteration to keep the GPU at
about `gpu_duty` utilisation, by sleeping in proportion to the work done.

Defaults come from env vars so every script shares one policy:
  SF_CPU_FRAC (0.6)  SF_GPU_MEM_FRAC (0.7)  SF_GPU_DUTY (0.65)
On Windows, unset vars fall back to the user's registry environment (HKCU\\Environment):
jobs started via WMI (tools/detached.ps1) may not inherit freshly set user env vars.
"""
from __future__ import annotations

import os
import sys
import time


def _setting(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None and sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                val = str(winreg.QueryValueEx(key, name)[0])
        except OSError:
            val = None
    return float(val) if val else default


CPU_FRAC = _setting("SF_CPU_FRAC", 0.6)
GPU_MEM_FRAC = _setting("SF_GPU_MEM_FRAC", 0.7)
GPU_DUTY = _setting("SF_GPU_DUTY", 0.65)


def cpu_budget() -> int:
    """Threads this job may use (at least 1)."""
    return max(1, int((os.cpu_count() or 2) * CPU_FRAC))


def worker_budget(requested: int) -> int:
    """DataLoader workers: the requested number, capped so workers + main thread fit the budget."""
    return max(0, min(requested, cpu_budget() - 2))


def _lower_priority() -> None:
    if sys.platform == "win32":
        import ctypes

        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        k32 = ctypes.windll.kernel32
        # HANDLE is pointer-sized: without explicit types ctypes truncates it to a
        # 32-bit int and the call silently fails.
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass.restype = ctypes.c_int
        if not k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS):
            print("[resources] could not lower process priority", file=sys.stderr)
    else:
        try:
            os.nice(10)
        except OSError:
            pass


def limit_resources(gpu: bool = True, threads: int | None = None) -> None:
    n = threads or cpu_budget()
    print(f"[resources] threads {n}/{os.cpu_count()}, gpu mem frac {GPU_MEM_FRAC}, gpu duty {GPU_DUTY}",
          file=sys.stderr)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    _lower_priority()  # inherited by DataLoader / ProcessPool workers
    try:
        import cv2

        cv2.setNumThreads(max(1, n // 2))
    except ImportError:
        pass
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(n)
    except ImportError:
        pass
    try:
        import torch

        torch.set_num_threads(n)
        if gpu and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRAC)
    except ImportError:
        pass


def worker_init(_worker_id: int) -> None:
    """DataLoader worker_init_fn: one OpenCV/torch thread per worker."""
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)


class GpuThrottle:
    """Sleep after each GPU step so the GPU is busy only ~`duty` of the time.

    Call begin() right before the GPU work and step() right after it: only that
    span counts as busy time. (Measuring from the previous step() instead also
    counted unrelated CPU work and other models in between, and slept for
    seconds after every stage change; that inflated inference ~2x.)"""

    def __init__(self, duty: float = GPU_DUTY):
        self.duty = min(1.0, max(0.05, duty))
        self.t = None

    def begin(self) -> None:
        self.t = time.perf_counter()

    def step(self, sync: bool = True) -> None:
        if self.duty >= 1.0 or self.t is None:
            return
        if sync:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        busy = time.perf_counter() - self.t
        time.sleep(busy * (1.0 / self.duty - 1.0))
        self.t = None
