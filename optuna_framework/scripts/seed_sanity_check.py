"""Manual seed sanity check runner.

Required manual patch for ``eg-torch/model.py`` before executing real runs:

```diff
diff --git a/eg-torch/model.py b/eg-torch/model.py
--- a/eg-torch/model.py
+++ b/eg-torch/model.py
@@
+import random
+import numpy as np
@@
 class ResearchModel:
     def __init__(self, config: dict[str, Any]):
@@
         self.loss_fn = TrainLoss()
+        self.seed = int(config.get("seed", 42))
@@
     def fit(self, dataset):
+        torch.manual_seed(self.seed)
+        np.random.seed(self.seed)
+        random.seed(self.seed)
+        if torch.cuda.is_available():
+            torch.cuda.manual_seed_all(self.seed)
+            torch.backends.cudnn.benchmark = False
+            torch.backends.cudnn.deterministic = True
         self.trainii = dataset.validinsts.detach().cpu().to(torch.long).clone()
@@
-        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
+        g = torch.Generator()
+        g.manual_seed(self.seed)
+        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=g)
```
"""

from __future__ import annotations

import argparse

if __package__ in (None, ""):
    from _script_common import bootstrap_repo_imports

    bootstrap_repo_imports()

from optuna_framework.config_renderer import render_config
from optuna_framework.paths import build_named_run_paths, resolve_study_root
from optuna_framework.runner import build_run_command, run_segment
from optuna_framework.scripts._script_common import adapter_for_name, print_command
from optuna_framework.studies.eg_torch_v1 import STUDY_SPEC


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run two same-seed baseline seg01 checks.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running runCombo.py")
    parser.add_argument("--study-root", default=None, help="Override default study root")
    parser.add_argument("--seed", type=int, default=42, help="Seed to check")
    return parser.parse_args()


def main() -> None:
    """Run or print the same-seed sanity check."""

    args = parse_args()
    study_root = resolve_study_root(args.study_root, STUDY_SPEC.name)
    adapter = adapter_for_name(STUDY_SPEC.adapter_name)
    segment = STUDY_SPEC.segment_by_name("seg01")
    params = adapter.baseline_params()
    run_paths_list = []
    for repeat in (1, 2):
        segment_dir = study_root / "seed_sanity" / f"seed_{args.seed}_repeat_{repeat}" / segment.name
        run_paths = build_named_run_paths(
            study_root,
            segment_dir,
            segment,
            snaptime=f"seed_sanity_seed_{args.seed}_repeat_{repeat}_{segment.name}",
            kind="seed_sanity",
        )
        run_paths_list.append(run_paths)

    if args.dry_run:
        print(f"[DRY-RUN] seed sanity study_root={study_root} seed={args.seed}")
        for idx, run_paths in enumerate(run_paths_list, start=1):
            print_command(f"[DRY-RUN] seed_sanity/repeat_{idx}", run_paths.config_path, build_run_command(run_paths.config_path))
        return

    metrics = []
    for run_paths in run_paths_list:
        overrides = {**STUDY_SPEC.fixed_overrides, "combo.model.seed": args.seed}
        render_config(STUDY_SPEC.baseline_config_path, run_paths, adapter, params, overrides)
        metrics.append(run_segment(run_paths))
    diff = abs(metrics[0].sharpe_idx - metrics[1].sharpe_idx)
    print(f"repeat_1_sharpe_idx={metrics[0].sharpe_idx:.8f}")
    print(f"repeat_2_sharpe_idx={metrics[1].sharpe_idx:.8f}")
    print(f"abs_diff={diff:.8f}")
    if diff > 1e-4:
        print("Warning: same-seed sharpe_idx diff is above 1e-4; manual review recommended.")


if __name__ == "__main__":
    main()

