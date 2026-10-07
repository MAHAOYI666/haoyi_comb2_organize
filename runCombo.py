"""Backward-compatible runCombo entry point; implementation lives in combo2."""
import sys
from combo2.cli import run_combo as _cli
from combo2.cli.run_combo import *

if __name__ == "__main__":
    raise SystemExit(main())
else:
    sys.modules[__name__] = _cli
