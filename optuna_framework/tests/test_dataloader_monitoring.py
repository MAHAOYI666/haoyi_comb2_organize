"""Static checks for DataLoader monitor API usage."""

from __future__ import annotations

from optuna_framework.paths import get_repo_root


UNSUPPORTED_MONITOR_SNIPPETS = (
    "monitor.accumulator(",
    ".tick(",
    ".flush(",
    "monitor.detail_enabled(",
    "monitor.maybe_section(",
)


def test_dataloader_uses_only_supported_perf_monitor_api() -> None:
    """DataLoader must not call accumulator APIs absent from PerfMonitor."""

    source = (get_repo_root() / "vendor" / "comb2" / "src" / "DataLoader.py").read_text(encoding="utf-8")

    for snippet in UNSUPPORTED_MONITOR_SNIPPETS:
        assert snippet not in source
