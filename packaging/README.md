# Protected Wheel Build

Build a Cython-based wheel that hides implementation modules as native `.so`
extensions while keeping only minimal package `__init__.py` wrappers.

Dry run:

```bash
../python310fs/bin/python packaging/build_protected_wheel.py --dry-run
```

Build:

```bash
../python310fs/bin/python packaging/build_protected_wheel.py
```

Download artifact:

```text
dist_protected/*.whl
```

The build compiles:

- `config`, `runCombo`, `runAblationByZero`, `runPosCorr`
- `comb2_simbase`
- `optuna_framework`
- `comb_eval`
- vendor packages: `src`, `comb2`, `comb2_pcmaster`, `comb2_metrics`
- `vendor.perf_monitor`

The script verifies that the resulting wheel does not contain protected `.py`
sources, except for minimal package `__init__.py` files needed for reliable
Python package imports.

Requirements:

- Python 3.10 ABI compatible with the deployment target
- `Cython`, `setuptools`, and `wheel`
- a C compiler available as `gcc` or `cc`
