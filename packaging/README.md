# Protected Wheel Build

Build a Cython-based wheel that hides implementation modules as native `.so`
extensions while keeping only minimal package `__init__.py` wrappers.

Dry run:

```bash
python packaging/build_protected_wheel.py --python python3.13 --dry-run
```

Build:

```bash
python packaging/build_protected_wheel.py --python python3.13
```

Download artifact:

```text
dist_protected/*.whl
```

Install the wheel in a Python 3.13 environment:

```bash
python -m pip install dist_protected/comb2_organize-0.1.0-cp313-cp313-linux_x86_64.whl
```

Run after installation:

```bash
runCombo config.xml
runEval config.xml
```

The build uses Python 3.13 for the wheel ABI and writes pinned third-party
dependency metadata that is compatible with Python 3.13 Linux x86_64 wheels.
The numpy pin follows `../aresium/pdm.lock`; pandas and pyarrow follow the
Python 3.13 dependency floor used by `../aressignalclient/pyproject.toml`. If
`python3.13` is not on `PATH`, pass an absolute path with `--python`.

The resulting wheel contains all local comb2_organize runtime code. Third-party
packages such as torch, LightGBM, pandas, numpy, pyarrow, matplotlib, Optuna,
psutil, and Plotly are not bundled into the wheel; they are declared in the
wheel metadata so `pip install` can resolve and install them for the target
Python 3.13 environment.

The build compiles:

- `config`, `runCombo`, `runEval`, `comboRunner`, `runAblationByZero`, `runPosCorr`
- `vendor/comb2-simbase` (`comb2_simbase` import package)
- `optuna_framework`
- `comb_eval`
- vendor packages: `src`, `comb2`, `comb2_pcmaster`, `comb2_metrics`
- `vendor.perf_monitor`

The script verifies that the resulting wheel does not contain protected `.py`
sources, except for minimal package `__init__.py` files needed for reliable
Python package imports.

Requirements:

- Python 3.13 ABI compatible with the deployment target
- `Cython`, `setuptools`, and `wheel` are declared in the generated build metadata
- a C compiler available as `gcc` or `cc`

Installed commands include:

- `combo-hello-world`
- `runCombo`
- `runEval`
- `comb-run`
- `comb-eval`
- `comb-combo-runner`
- `comb-ablation-zero`
- `comb-pos-corr`
