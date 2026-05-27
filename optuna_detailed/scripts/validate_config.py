"""Validate detailed Optuna XML config and write config_plan.txt."""

from __future__ import annotations

import argparse
import sys

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_detailed.config_validator import build_validation_plan, write_config_plan
from optuna_detailed.scripts._script_common import add_common_config_args, load_config_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate optuna_detailed config.xml.")
    add_common_config_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_from_args(args)
    plan = build_validation_plan(config)
    path = write_config_plan(plan)
    print(f"config_plan={path}")
    print(f"status={plan.status}")
    print(f"plan_hash={plan.plan_hash}")
    if not plan.is_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
