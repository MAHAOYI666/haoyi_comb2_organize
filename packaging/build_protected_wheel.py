from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

PACKAGE_SOURCES = {
    "optuna_framework": REPO_ROOT / "optuna_framework",
    "comb_eval": REPO_ROOT / "evals" / "comb_eval",
    "src": REPO_ROOT / "vendor" / "comb2" / "src",
    "comb2": REPO_ROOT / "vendor" / "comb2" / "comb2",
    "comb2_pcmaster": REPO_ROOT / "vendor" / "comb2-pcmaster" / "comb2_pcmaster",
    "comb2_metrics": REPO_ROOT / "vendor" / "comb2-metrics" / "comb2_metrics",
}

MODULE_SOURCES = {
    "config": REPO_ROOT / "config.py",
    "runCombo": REPO_ROOT / "runCombo.py",
    "runAblationByZero": REPO_ROOT / "runAblationByZero.py",
    "runPosCorr": REPO_ROOT / "runPosCorr.py",
    "vendor.perf_monitor": REPO_ROOT / "vendor" / "perf_monitor.py",
}

ENTRY_POINTS = {
    "comb-run": "runCombo:main",
    "comb-ablation-zero": "runAblationByZero:main",
    "comb-pos-corr": "runPosCorr:main",
    "comb-eval": "comb_eval.cli:main",
}
CONSOLE_SCRIPTS = [f"{name}={target}" for name, target in ENTRY_POINTS.items()]

IGNORED_DIRS = {"__pycache__", ".pytest_cache", "tests", "studies"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib", ".c", ".cpp"}
ALLOWED_SOURCE_FILES = {
    "comb2/__init__.py",
    "comb2_metrics/__init__.py",
    "comb2_pcmaster/__init__.py",
    "comb_eval/__init__.py",
    "optuna_framework/__init__.py",
    "optuna_framework/scripts/__init__.py",
    "src/__init__.py",
    "src/codec/__init__.py",
    "vendor/__init__.py",
}


def main() -> None:
    args = parse_args()
    python = args.python.expanduser().resolve()
    build_root = args.build_root.resolve()
    stage_root = build_root / "protected_src"
    dist_dir = args.dist_dir.resolve()

    if build_root.exists() and not args.no_clean:
        shutil.rmtree(build_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    prepare_stage(stage_root)
    write_build_files(stage_root, args.name, args.version)

    if args.dry_run:
        print_plan(stage_root)
        return

    require_compiler()
    subprocess.run(
        [str(python), "setup.py", "bdist_wheel", "--dist-dir", str(dist_dir)],
        cwd=stage_root,
        check=True,
    )
    wheels = sorted(dist_dir.glob(f"{normalize_dist_name(args.name)}-*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        raise RuntimeError(f"no wheel produced in {dist_dir}")
    wheel = wheels[-1]
    verify_wheel(wheel)
    print(f"built protected wheel: {wheel}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Cython-protected wheel for comb2_organize and bundled vendor packages.")
    parser.add_argument("--python", type=Path, default=Path("../python310fs/bin/python"), help="Python executable used to build the wheel")
    parser.add_argument("--name", default="comb2-organize-protected", help="Wheel distribution name")
    parser.add_argument("--version", default="0.1.0", help="Wheel version")
    parser.add_argument("--build-root", type=Path, default=REPO_ROOT / "build" / "protected_wheel", help="Temporary build directory")
    parser.add_argument("--dist-dir", type=Path, default=REPO_ROOT / "dist_protected", help="Output wheel directory")
    parser.add_argument("--dry-run", action="store_true", help="Prepare the build tree and print what would be compiled")
    parser.add_argument("--no-clean", action="store_true", help="Reuse the existing build root")
    return parser.parse_args()


def prepare_stage(stage_root: Path) -> None:
    for package, source in PACKAGE_SOURCES.items():
        copy_tree(source, stage_root / package)

    for module, source in MODULE_SOURCES.items():
        target = stage_root / Path(*module.split(".")).with_suffix(".py")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    vendor_init = stage_root / "vendor" / "__init__.py"
    vendor_init.parent.mkdir(parents=True, exist_ok=True)
    vendor_init.write_text('"""Vendor support package for compiled comb2_organize wheels."""\n', encoding="utf-8")


def copy_tree(source: Path, target: Path) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in IGNORED_DIRS}
        ignored.update(name for name in names if Path(name).suffix in IGNORED_SUFFIXES)
        return ignored

    shutil.copytree(source, target, ignore=ignore, dirs_exist_ok=True)


def write_build_files(stage_root: Path, name: str, version: str) -> None:
    extensions = extension_specs(stage_root)
    packages = package_names(stage_root)
    setup_py = f"""
from __future__ import annotations

from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [(pkg, mod, file) for pkg, mod, file in modules if mod == "__init__"]


extensions = [
{format_extensions(extensions)}
]

setup(
    name={name!r},
    version={version!r},
    description="Protected binary wheel for comb2_organize",
    python_requires=">=3.10",
    packages={packages!r},
    ext_modules=cythonize(
        extensions,
        compiler_directives={{
            "language_level": "3",
            "embedsignature": False,
        }},
        annotate=False,
    ),
    cmdclass={{"build_py": build_py}},
    entry_points={{"console_scripts": {CONSOLE_SCRIPTS!r}}},
    install_requires=[
        "numpy",
        "pandas",
        "pyarrow",
        "torch",
        "optuna",
        "matplotlib",
    ],
    zip_safe=False,
)
"""
    (stage_root / "setup.py").write_text(textwrap.dedent(setup_py).lstrip(), encoding="utf-8")
    (stage_root / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [build-system]
            requires = ["setuptools>=68", "wheel", "Cython>=3.0"]
            build-backend = "setuptools.build_meta"
            """
        ).lstrip(),
        encoding="utf-8",
    )


def extension_specs(stage_root: Path) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for package in PACKAGE_SOURCES:
        package_root = stage_root / package
        for path in sorted(package_root.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            module = ".".join(path.relative_to(stage_root).with_suffix("").parts)
            specs.append((module, path.relative_to(stage_root).as_posix()))

    for module in sorted(MODULE_SOURCES):
        path = stage_root / Path(*module.split(".")).with_suffix(".py")
        specs.append((module, path.relative_to(stage_root).as_posix()))
    return specs


def package_names(stage_root: Path) -> list[str]:
    packages = []
    for init_file in sorted(stage_root.rglob("__init__.py")):
        packages.append(".".join(init_file.parent.relative_to(stage_root).parts))
    return packages


def format_extensions(extensions: list[tuple[str, str]]) -> str:
    return "\n".join(f"    Extension({module!r}, [{path!r}])," for module, path in extensions)


def require_compiler() -> None:
    if shutil.which("gcc") or shutil.which("cc"):
        return
    raise RuntimeError("Cython extension build requires gcc or cc, but no compiler was found in PATH.")


def verify_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        source_files = sorted(name for name in archive.namelist() if name.endswith(".py"))
    disallowed = [name for name in source_files if name not in ALLOWED_SOURCE_FILES]
    if disallowed:
        joined = "\n".join(f"  {name}" for name in disallowed)
        raise RuntimeError(f"protected wheel still contains source files:\n{joined}")


def print_plan(stage_root: Path) -> None:
    extensions = extension_specs(stage_root)
    packages = package_names(stage_root)
    print(f"staged source: {stage_root}")
    print(f"packages: {len(packages)}")
    print(f"compiled extensions: {len(extensions)}")
    for module, path in extensions:
        print(f"  {module}: {path}")


def normalize_dist_name(name: str) -> str:
    return name.replace("-", "_").replace(".", "_")


if __name__ == "__main__":
    main()
