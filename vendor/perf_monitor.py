from __future__ import annotations

import csv
import inspect
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable


@dataclass
class PerfMonitorConfig:
    enabled: bool = False
    output_path: str | None = None
    format: str = "csv"
    print_summary: bool = True
    collect_gpu: bool = True
    sync_cuda: bool = False
    verbose: bool = False


class PerfMonitor:
    FIELDNAMES = [
        "event",
        "date",
        "start_time",
        "end_time",
        "wall_s",
        "cpu_s",
        "rss_mb",
        "rss_delta_mb",
        "gpu_allocated_mb",
        "gpu_reserved_mb",
        "gpu_device",
        "success",
        "error",
    ]

    def __init__(self, config: PerfMonitorConfig):
        self.config = config
        self.enabled = bool(config.enabled)
        self.output_path = Path(config.output_path).expanduser().resolve() if config.output_path else None
        self._started_at = time.perf_counter()
        self._closed = False
        self._writer = None
        self._file = None
        self._torch = None
        self._gpu_device = ""

        if not self.enabled:
            return
        if self.config.format != "csv":
            raise ValueError(f"unsupported perf monitor format: {self.config.format}")
        if self.output_path is None:
            raise ValueError("perf monitor output_path is required when monitor is enabled")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._init_gpu()
        print(f"[PERF] enabled output={self.output_path}")

    @classmethod
    def from_config(cls, organize_config: dict[str, Any]) -> "PerfMonitor":
        monitor_config = dict(organize_config.get("monitor", {}))
        if monitor_config.get("enabled") and not monitor_config.get("output_path"):
            monitor_config["output_path"] = str(Path(organize_config["constants"]["output_root"]) / "perf_metrics.csv")
        return cls(PerfMonitorConfig(**monitor_config))

    def section(self, event: str, date: int | None = None):
        if not self.enabled:
            return nullcontext()
        return self._section(event, date)

    def decorate(self, event: str, date_arg: str | None = None):
        def decorator(func: Callable):
            signature = inspect.signature(func)

            @wraps(func)
            def wrapper(*args, **kwargs):
                date = None
                if date_arg:
                    bound = signature.bind_partial(*args, **kwargs)
                    date = bound.arguments.get(date_arg)
                with self.section(event, date=date):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def patch_method(self, cls: type, method_name: str, event: str, date_arg: str | None = None):
        original = getattr(cls, method_name)
        if getattr(original, "_perf_patched", False):
            return
        decorated = self.decorate(event, date_arg=date_arg)(original)
        decorated._perf_patched = True
        setattr(cls, method_name, decorated)

    @contextmanager
    def _section(self, event: str, date: int | None = None):
        self._sync_cuda_if_needed()
        start_time = datetime.now().isoformat(timespec="seconds")
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        start_rss = self._read_rss_mb()
        success = True
        error = ""
        try:
            yield
        except Exception as exc:
            success = False
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._sync_cuda_if_needed()
            end_wall = time.perf_counter()
            end_cpu = time.process_time()
            end_rss = self._read_rss_mb()
            gpu = self._read_gpu_memory()
            self._write_row(
                {
                    "event": event,
                    "date": "" if date is None else int(date),
                    "start_time": start_time,
                    "end_time": datetime.now().isoformat(timespec="seconds"),
                    "wall_s": f"{end_wall - start_wall:.6f}",
                    "cpu_s": f"{end_cpu - start_cpu:.6f}",
                    "rss_mb": self._format_float(end_rss),
                    "rss_delta_mb": self._format_float(end_rss - start_rss if end_rss is not None and start_rss is not None else None),
                    "gpu_allocated_mb": self._format_float(gpu.get("allocated_mb")),
                    "gpu_reserved_mb": self._format_float(gpu.get("reserved_mb")),
                    "gpu_device": gpu.get("device", ""),
                    "success": str(success).lower(),
                    "error": error,
                }
            )

    def close(self):
        if not self.enabled or self._closed:
            return
        self._closed = True
        total_wall_s = time.perf_counter() - self._started_at
        if self._file is not None:
            self._file.flush()
            self._file.close()
        if self.config.print_summary:
            print(f"[PERF] summary output={self.output_path} total_wall_s={total_wall_s:.3f}")

    def _write_row(self, row: dict[str, Any]):
        if self._writer is None or self._file is None:
            return
        self._writer.writerow(row)
        self._file.flush()

    def _init_gpu(self):
        if not self.config.collect_gpu:
            return
        try:
            import torch
        except Exception:
            return
        if not torch.cuda.is_available():
            return
        self._torch = torch
        device = torch.cuda.current_device()
        self._gpu_device = f"cuda:{device} {torch.cuda.get_device_name(device)}"

    def _sync_cuda_if_needed(self):
        if self._torch is not None and self.config.sync_cuda:
            self._torch.cuda.synchronize()

    def _read_gpu_memory(self) -> dict[str, Any]:
        if self._torch is None:
            return {}
        return {
            "allocated_mb": self._torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved_mb": self._torch.cuda.memory_reserved() / 1024 / 1024,
            "device": self._gpu_device,
        }

    @staticmethod
    def _read_rss_mb() -> float | None:
        status_path = Path("/proc/self/status")
        try:
            for line in status_path.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024
        except OSError:
            return None
        return None



def current_rss_mb() -> float:
    rss = PerfMonitor._read_rss_mb()
    return 0.0 if rss is None else rss


def progress_line(stage: str, current: int, total: int, start_time: float, details: str = "") -> str:
    suffix = f" ({details})" if details else ""
    return (
        f"[INFO|{time.strftime('%H:%M:%S', time.localtime())}] {stage}--"
        f"updates {current}/{total} mem {current_rss_mb():.2f} MB--"
        f"{time.perf_counter() - start_time:.2f} secs{suffix}"
    )


def print_progress(stage: str, current: int, total: int, start_time: float, details: str = "", final: bool = False):
    print(progress_line(stage, current, total, start_time, details), end="\n" if final else "\r", flush=True)
