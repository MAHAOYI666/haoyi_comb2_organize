from __future__ import annotations

import argparse
import importlib.machinery
import os
import sys
from pathlib import Path


def _has_runtime_module(root: Path, name: str) -> bool:
    if (root / f"{name}.py").is_file():
        return True
    return importlib.machinery.PathFinder.find_spec(name, [str(root)]) is not None


def _has_runtime_root(root: Path) -> bool:
    return _has_runtime_module(root, "runCombo") and _has_runtime_module(root, "config")


def _find_repo_root(start: Path, explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_root = os.environ.get("COMB2_ORGANIZE_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend([start, *start.parents])
    for parent in [start, *start.parents]:
        candidates.append(parent / "comb2_organize")

    for candidate in candidates:
        root = candidate.resolve()
        if _has_runtime_root(root):
            return root
    raise FileNotFoundError(
        "cannot locate comb2_organize root; pass --repo-root or set COMB2_ORGANIZE_ROOT"
    )


def _resolve_config(script_dir: Path, positional: str | None, flag_value: str | None) -> Path:
    raw = flag_value or positional
    if raw:
        return Path(raw).expanduser().resolve()

    default_config = script_dir / "config.xml"
    if default_config.is_file():
        return default_config.resolve()

    cwd_config = Path.cwd() / "config.xml"
    if cwd_config.is_file():
        return cwd_config.resolve()

    raise FileNotFoundError(
        "config.xml not found; pass a config path or run from a directory containing config.xml"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="comb-combo-runner",
        description="Run a comb2 research model through the standard runCombo pipeline.",
        epilog="example: comb-combo-runner config.xml --repo-root /path/to/comb2_organize",
    )
    parser.add_argument("config", nargs="?", default=None, help="Path to XML experiment config")
    parser.add_argument("--config", dest="config_flag", default=None, help="Path to XML experiment config")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Path to comb2_organize; defaults to COMB2_ORGANIZE_ROOT or a parent containing runCombo.py",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    try:
        config_path = _resolve_config(script_dir, args.config, args.config_flag)
        repo_root = _find_repo_root(script_dir, args.repo_root)
    except FileNotFoundError as exc:
        print(f"comb-combo-runner: {exc}", file=sys.stderr)
        return 2

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import runCombo

    old_argv = sys.argv[:]
    try:
        sys.argv = [str(repo_root / "runCombo.py"), str(config_path)]
        return int(runCombo.main() or 0)
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
