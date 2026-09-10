from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import base64
import os
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = REPO_ROOT / "VERSION"
DISTRIBUTION_NAME = "combo2"

PACKAGE_SOURCES = {
    "comb2_templates": REPO_ROOT / "comb2_templates",
    "comb2_simbase": REPO_ROOT / "vendor" / "comb2-simbase" / "comb2_simbase",
    "optuna_framework": REPO_ROOT / "optuna_framework",
    "comb_eval": REPO_ROOT / "evals" / "comb_eval",
    "comb2": REPO_ROOT / "vendor" / "comb2" / "comb2",
    "comb2_pcmaster": REPO_ROOT / "vendor" / "comb2-pcmaster" / "comb2_pcmaster",
    "comb2_metrics": REPO_ROOT / "vendor" / "comb2-metrics" / "comb2_metrics",
}

MODULE_SOURCES = {
    "config": REPO_ROOT / "config.py",
    "runCombo": REPO_ROOT / "runCombo.py",
    "runEval": REPO_ROOT / "runEval.py",
    "comboRunner": REPO_ROOT / "comboRunner.py",
    "runPosCorr": REPO_ROOT / "runPosCorr.py",
    "comboHelloWorld": REPO_ROOT / "comboHelloWorld.py",
    "vendor.perf_monitor": REPO_ROOT / "vendor" / "perf_monitor.py",
}

ENTRY_POINTS = {
    "runCombo": "runCombo:main",
    "runEval": "runEval:main",
    "comb-run": "runCombo:main",
    "comb-combo-runner": "comboRunner:main",
    "comb-pos-corr": "runPosCorr:main",
    "comb-eval": "comb_eval.cli:main",
    "combo-hello-world": "comboHelloWorld:main",
}
CONSOLE_SCRIPTS = [f"{name}={target}" for name, target in ENTRY_POINTS.items()]

# These pins target Python 3.13 Linux x86_64 wheels.  numpy follows
# ../aresium/pdm.lock, while pandas/pyarrow follow the lower bounds in
# ../aressignalclient/pyproject.toml.
BUILD_DEPENDENCY_PINS = (
    ("setuptools", "82.0.1"),
    ("wheel", "0.47.0"),
    ("Cython", "3.0.12"),
)
RUNTIME_DEPENDENCY_PINS = (
    ("numpy", "2.3.5"),
    ("pandas", "3.0.2"),
    ("pyarrow", "23.0.1"),
    ("torch", "2.9.1"),
    ("matplotlib", "3.9.4"),
    ("optuna", "4.8.0"),
    ("psutil", "7.2.2"),
    ("plotly", "6.7.0"),
    ("lightgbm", "4.4.0"),
    ("Mosek", "11.0.25"),
)

IGNORED_DIRS = {"__pycache__", ".pytest_cache", "tests", "studies"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib", ".c", ".cpp"}
ALLOWED_SOURCE_FILES = {
    "comb2/__init__.py",
    "comb2/codec/__init__.py",
    "comb2_templates/__init__.py",
    "comb2_simbase/__init__.py",
    "comb2_pcmaster/default_strategy.py",
    "comb2_metrics/__init__.py",
    "comb2_pcmaster/__init__.py",
    "comb_eval/__init__.py",
    "optuna_framework/__init__.py",
    "optuna_framework/scripts/__init__.py",
    "vendor/__init__.py",
}
PLAIN_SOURCE_FILES = {"comb2_pcmaster/default_strategy.py"}


def main() -> None:
    args = parse_args()
    version = read_version()
    python = resolve_path_arg(args.python)
    assert_python_313(python)
    dependencies = resolve_dependencies()
    build_root = args.build_root.resolve()
    stage_root = build_root / "protected_src"
    dist_dir = args.dist_dir.resolve()

    lock_path = REPO_ROOT / "build" / ".protected_wheel.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        build_wheel(args, version, python, dependencies, build_root, stage_root, dist_dir)


def build_wheel(
    args: argparse.Namespace,
    version: str,
    python: Path,
    dependencies: dict[str, object],
    build_root: Path,
    stage_root: Path,
    dist_dir: Path,
) -> None:
    if build_root.exists() and not args.no_clean:
        shutil.rmtree(build_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    prepare_stage(stage_root)
    write_build_files(stage_root, DISTRIBUTION_NAME, version, dependencies)

    if args.dry_run:
        print_plan(stage_root, dependencies, python)
        return

    require_compiler()
    ensure_pip(python)
    command = [str(python), "-m", "pip", "wheel", ".", "--no-deps", "--wheel-dir", str(dist_dir)]
    if args.no_build_isolation:
        command.append("--no-build-isolation")
    subprocess.run(command, cwd=stage_root, check=True)
    wheels = sorted(
        [
            path
            for path in dist_dir.glob("*.whl")
            if path.name.startswith(f"{DISTRIBUTION_NAME}-")
        ],
        key=lambda path: path.stat().st_mtime,
    )
    if not wheels:
        raise RuntimeError(f"no wheel produced in {dist_dir}")
    wheel = wheels[-1]
    strip_wheel_extensions(wheel)
    verify_wheel(wheel)
    print(f"built protected wheel: {wheel}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Cython-protected combo2 wheel and bundled runtime packages.")
    parser.add_argument("--python", type=Path, default=default_build_python(), help="Python 3.13 executable used to build the wheel")
    parser.add_argument("--build-root", type=Path, default=REPO_ROOT / "build" / "protected_wheel", help="Temporary build directory")
    parser.add_argument("--dist-dir", type=Path, default=REPO_ROOT / "dist_protected", help="Output wheel directory")
    parser.add_argument("--dry-run", action="store_true", help="Prepare the build tree and print what would be compiled")
    parser.add_argument("--no-clean", action="store_true", help="Reuse the existing build root")
    parser.add_argument("--no-build-isolation", action="store_true", help="Build with packages already installed in --python")
    return parser.parse_args()


def read_version() -> str:
    version = VERSION_FILE.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError(f"empty version file: {VERSION_FILE}")
    return version


def default_build_python() -> Path:
    python313 = shutil.which("python3.13")
    if python313:
        return Path(python313)
    repo_venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if repo_venv_python.exists():
        return repo_venv_python
    return Path(sys.executable)


def resolve_path_arg(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    if len(expanded.parts) == 1:
        executable = shutil.which(str(expanded))
        if executable:
            return Path(executable).resolve()
    return (REPO_ROOT / expanded).resolve()


def assert_python_313(python: Path) -> None:
    proc = subprocess.run(
        [str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    version = proc.stdout.strip()
    if version != "3.13":
        raise RuntimeError(f"protected wheel must be built with Python 3.13, got Python {version} from {python}")


def ensure_pip(python: Path) -> None:
    probe = subprocess.run([str(python), "-m", "pip", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode == 0:
        return
    subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], check=True)


def resolve_dependencies() -> dict[str, object]:
    return {
        "build_requires": pin_dependencies(BUILD_DEPENDENCY_PINS),
        "install_requires": pin_dependencies(RUNTIME_DEPENDENCY_PINS),
    }


def pin_dependencies(pins: tuple[tuple[str, str], ...]) -> list[str]:
    return [f"{name}=={version}" for name, version in pins]


def prepare_stage(stage_root: Path) -> None:
    for package, source in PACKAGE_SOURCES.items():
        copy_tree(source, stage_root / package)

    shutil.copy2(REPO_ROOT / "config.human", stage_root / "comb2_templates" / "config.human")

    for module, source in MODULE_SOURCES.items():
        target = stage_root / Path(*module.split(".")).with_suffix(".py")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    vendor_init = stage_root / "vendor" / "__init__.py"
    vendor_init.parent.mkdir(parents=True, exist_ok=True)
    vendor_init.write_text('"""Vendor support package for compiled Combo2 wheels."""\n', encoding="utf-8")


def copy_tree(source: Path, target: Path) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in IGNORED_DIRS}
        ignored.update(name for name in names if Path(name).suffix in IGNORED_SUFFIXES)
        return ignored

    shutil.copytree(source, target, ignore=ignore, dirs_exist_ok=True)


def write_build_files(stage_root: Path, name: str, version: str, dependencies: dict[str, object]) -> None:
    extensions = extension_specs(stage_root)
    packages = package_names(stage_root)
    install_requires = dependencies["install_requires"]
    build_requires = dependencies["build_requires"]
    setup_py = f"""
from __future__ import annotations

from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [(pkg, mod, file) for pkg, mod, file in modules if mod == "__init__"]


class build_ext(_build_ext):
    def _ensure_extension_dirs(self, extensions):
        for ext in extensions:
            for source in ext.sources:
                (Path(self.build_temp) / Path(source).parent).mkdir(parents=True, exist_ok=True)

    def _wrap_compiler(self):
        if getattr(self.compiler, "_comb2_dir_wrapper", False):
            return

        original_compile = self.compiler._compile

        def compile_with_dirs(obj, src, ext, cc_args, extra_postargs, pp_opts):
            Path(obj).parent.mkdir(parents=True, exist_ok=True)
            return original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = compile_with_dirs

        original_link_shared_object = self.compiler.link_shared_object

        def link_shared_object_with_dirs(objects, output_libname, *args, **kwargs):
            Path(output_libname).parent.mkdir(parents=True, exist_ok=True)
            return original_link_shared_object(objects, output_libname, *args, **kwargs)

        self.compiler.link_shared_object = link_shared_object_with_dirs
        self.compiler._comb2_dir_wrapper = True

    def build_extensions(self):
        self._wrap_compiler()
        self._ensure_extension_dirs(self.extensions)
        super().build_extensions()

    def build_extension(self, ext):
        self._ensure_extension_dirs([ext])
        super().build_extension(ext)


extensions = [
{format_extensions(extensions)}
]

setup(
    name={name!r},
    version={version!r},
    description="Protected binary wheel for Combo2",
    python_requires=">=3.13,<3.14",
    packages={packages!r},
    ext_modules=cythonize(
        extensions,
        compiler_directives={{
            "language_level": "3",
            "embedsignature": False,
        }},
        annotate=False,
    ),
    cmdclass={{"build_py": build_py, "build_ext": build_ext}},
    entry_points={{"console_scripts": {CONSOLE_SCRIPTS!r}}},
    install_requires={install_requires!r},
    package_data={{
        "comb2_templates": ["config.human"],
        "comb2_pcmaster": ["default_strategy.py"],
        "comb2_simbase": ["index_mask/memmap_mask/*.npy"],
        "optuna_framework": ["config.xml"],
    }},
    zip_safe=False,
)
"""
    (stage_root / "setup.py").write_text(textwrap.dedent(setup_py).lstrip(), encoding="utf-8")
    (stage_root / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""
            [build-system]
            requires = {build_requires!r}
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
            if path.relative_to(stage_root).as_posix() in PLAIN_SOURCE_FILES:
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
        if init_file.relative_to(stage_root).parts[0] not in {*PACKAGE_SOURCES, "vendor"}:
            continue
        packages.append(".".join(init_file.parent.relative_to(stage_root).parts))
    return packages


def format_extensions(extensions: list[tuple[str, str]]) -> str:
    return "\n".join(f"    Extension({module!r}, [{path!r}])," for module, path in extensions)


def require_compiler() -> None:
    if shutil.which("gcc") or shutil.which("cc"):
        return
    raise RuntimeError("Cython extension build requires gcc or cc, but no compiler was found in PATH.")


def require_strip() -> str:
    strip_bin = shutil.which("strip")
    if strip_bin:
        return strip_bin
    raise RuntimeError("protected wheel strip step requires `strip`, but it was not found in PATH.")


def strip_wheel_extensions(wheel: Path) -> None:
    strip_bin = require_strip()
    work_dir = wheel.parent / f".strip_{wheel.stem}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(work_dir)
        for so_path in sorted(work_dir.rglob("*.so")):
            subprocess.run([strip_bin, "--strip-unneeded", str(so_path)], check=True)
        rewrite_wheel_from_dir(work_dir, wheel)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def rewrite_wheel_from_dir(work_dir: Path, wheel: Path) -> None:
    dist_info_dirs = list(work_dir.glob("*.dist-info"))
    if len(dist_info_dirs) != 1:
        raise RuntimeError(f"expected exactly one .dist-info directory in {work_dir}, got {dist_info_dirs}")
    dist_info = dist_info_dirs[0]
    record_path = dist_info / "RECORD"
    entries: list[tuple[str, str, str]] = []
    if record_path.exists():
        record_path.unlink()
    for path in sorted(p for p in work_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(work_dir).as_posix()
        digest = hashlib.sha256(path.read_bytes()).digest()
        b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        size = str(path.stat().st_size)
        entries.append((rel, f"sha256={b64}", size))
    entries.append((record_path.relative_to(work_dir).as_posix(), "", ""))
    with record_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(entries)
    tmp_wheel = wheel.with_suffix(".tmp.whl")
    if tmp_wheel.exists():
        tmp_wheel.unlink()
    with zipfile.ZipFile(tmp_wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in work_dir.rglob("*") if p.is_file()):
            archive.write(path, arcname=path.relative_to(work_dir).as_posix())
    os.replace(tmp_wheel, wheel)


def verify_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        source_files = sorted(name for name in archive.namelist() if name.endswith(".py"))
    disallowed = [name for name in source_files if name not in ALLOWED_SOURCE_FILES]
    if disallowed:
        joined = "\n".join(f"  {name}" for name in disallowed)
        raise RuntimeError(f"protected wheel still contains source files:\n{joined}")


def print_plan(stage_root: Path, dependencies: dict[str, object], python: Path) -> None:
    extensions = extension_specs(stage_root)
    packages = package_names(stage_root)
    print(f"staged source: {stage_root}")
    print(f"build python: {python}")
    print("dependency version source: built-in Python 3.13 wheel-compatible pins")
    print(f"packages: {len(packages)}")
    print(f"compiled extensions: {len(extensions)}")
    print("install_requires:")
    for dep in dependencies["install_requires"]:
        print(f"  {dep}")
    print("build-system.requires:")
    for dep in dependencies["build_requires"]:
        print(f"  {dep}")
    for module, path in extensions:
        print(f"  {module}: {path}")


if __name__ == "__main__":
    main()
