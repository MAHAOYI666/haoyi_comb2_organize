from __future__ import annotations

import argparse
import importlib.resources as resources
import sys
import textwrap
from pathlib import Path


MODEL_TEMPLATE = r'''
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Model(nn.Module):
    def __init__(
        self,
        input_size: int,
        ts_days: int,
        hidden_size: int,
        fc_size: int,
        trainii: torch.Tensor,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.trainii = trainii
        self.dropout = nn.Dropout(dropout)
        self.Q = nn.Linear(input_size, hidden_size)
        self.K = nn.Linear(input_size, hidden_size)
        self.hdfc = nn.Linear(input_size * 6, hidden_size)
        self.mlpfc1 = nn.Linear(hidden_size, fc_size, bias=True)
        self.mlpfc2 = nn.Linear(fc_size, fc_size, bias=True)
        self.mlpfc3 = nn.Linear(fc_size, 1, bias=True)

    def attn(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3).contiguous()
        q = self.Q(x[:, :, -1])
        k = self.K(x[:, :, :-1])
        attn_scores = torch.matmul(q.unsqueeze(2), k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, x[:, :, :-1]).squeeze(2)

    def ts_slope(self, x: torch.Tensor) -> torch.Tensor:
        _, steps, _, _ = x.shape
        ts = torch.arange(steps, dtype=x.dtype, device=x.device)
        ts = ts - ts.mean()
        ts_var = ts.pow(2).sum()
        cov = (ts.view(1, steps, 1, 1) * (x - x.mean(dim=1, keepdim=True))).sum(dim=1)
        return cov / ts_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 4.0
        x_last = x[:, -1]
        x_tsmean = x.mean(dim=1)
        x_tsdelta = x[:, -1] - x[:, -2]
        x_tsslope = self.ts_slope(x)
        x_attn = self.attn(x)
        x = torch.cat([x_last, x_tsmean, x_last - x_tsmean, x_tsdelta, x_tsslope, x_attn], dim=-1)

        out = self.hdfc(x)
        out = self.dropout(out)
        out = F.leaky_relu(out, negative_slope=0.1)
        out = self.mlpfc1(out)
        out = self.dropout(out)
        out = F.leaky_relu(out, negative_slope=0.1)
        out = self.mlpfc2(out)
        out = self.dropout(out)
        out = F.leaky_relu(out, negative_slope=0.1)
        out = self.mlpfc3(out)
        return out.squeeze(-1)


class ICLoss(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = x * w
        y = y * w
        y = y - y.mean(dim=1, keepdim=True)
        x = x - x.mean(dim=1, keepdim=True)
        x = x * w
        y = y * w
        numerator = torch.sum(x * y, dim=1)
        denominator = torch.sqrt(torch.sum(x**2, dim=1) + 1e-8)
        return (1 - numerator / denominator).mean()


TrainLoss = ICLoss


class ResearchModel:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.dtype = config.get("dtype", torch.float16)
        self.ts_days = int(config.get("tsDays", 8))
        self.num_features = int(config.get("num_features", 1))
        self.device = torch.device(config.get("device", "cpu"))
        adaptive_hidden_size = max(64, self.num_features * 8)
        self.hidden_size = int(config.get("hidden_size", adaptive_hidden_size))
        self.fc_size = int(config.get("fc_size", self.hidden_size))
        self.dropout = float(config.get("dropout", 0.5))
        self.lr = float(config.get("lr", 2e-6))
        self.epochs = int(config.get("epochs", 15))
        self.batch_size = int(config.get("batch_size", 3))
        self.weight_decay = float(config.get("weight_decay", 1e-6))
        self.scheduler_step_size = int(config.get("scheduler_step_size", 10))
        self.scheduler_gamma = float(config.get("scheduler_gamma", 0.5))
        self.grad_clip = float(config.get("grad_clip", 10.0))
        self.early_stopping_patience = int(config.get("early_stopping_patience", 5))
        self.seed = int(config.get("seed", 42))
        self.trainii: torch.Tensor | None = None
        self.model: Model | None = None
        self.loss_fn = TrainLoss()
        self._apply_seed()

    def _apply_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def _dataloader_generator(self) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return generator

    def _init_model(self, trainii: torch.Tensor) -> Model:
        model = Model(
            input_size=self.num_features,
            ts_days=self.ts_days,
            hidden_size=self.hidden_size,
            fc_size=self.fc_size,
            trainii=trainii,
            dropout=self.dropout,
        )
        return model.to(self.device)

    def _next_batch(self, iterator):
        return next(iterator)

    def _batch_to_device(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
        return (
            x.to(self.device, dtype=torch.float32, non_blocking=True),
            y.to(self.device, dtype=torch.float32, non_blocking=True),
            w.to(self.device, dtype=torch.float32, non_blocking=True),
        )

    def _zero_grad(self, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def _forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_loss(self, pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, y, w)

    def _backward_loss(self, loss: torch.Tensor):
        loss.backward()

    def _clip_grad(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

    def _optimizer_step(self, optimizer):
        optimizer.step()

    def _loss_to_float(self, loss: torch.Tensor) -> float:
        return float(loss.detach().cpu())

    def fit(self, dataset):
        self._apply_seed()
        self.trainii = dataset.validinsts.detach().cpu().to(torch.long).clone()
        self.model = self._init_model(self.trainii)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self._dataloader_generator(),
            pin_memory=self.device.type == "cuda",
        )
        self.model.train()
        best_loss = float("inf")
        best_state_dict = None
        stale_epochs = 0

        for epoch in range(self.epochs):
            total_loss = 0.0
            batches = 0
            iterator = iter(dataloader)
            while True:
                try:
                    _, x, y, w = self._next_batch(iterator)
                except StopIteration:
                    break
                x, y, w = self._batch_to_device(x, y, w)
                self._zero_grad(optimizer)
                pred = self._forward_batch(x)
                loss = self._compute_loss(pred, y, w)
                self._backward_loss(loss)
                self._clip_grad()
                self._optimizer_step(optimizer)
                total_loss += self._loss_to_float(loss)
                batches += 1
            scheduler.step()
            avg_loss = total_loss / max(batches, 1)
            print(f"[FIT] epoch={epoch + 1}/{self.epochs} loss={avg_loss:.6f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state_dict = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.early_stopping_patience:
                    print(
                        f"[FIT] early stopping at epoch={epoch + 1}/{self.epochs} "
                        f"best_loss={best_loss:.6f}"
                    )
                    break
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
        return self

    @torch.no_grad()
    def predict(self, x_window: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ValueError("model is not fitted")
        if x_window.dim() != 3:
            raise ValueError(f"expected 3D feature tensor, got shape={tuple(x_window.shape)}")
        self.model.eval()
        x = x_window.unsqueeze(0).to(self.device, dtype=torch.float32)
        pred = self.model(x).squeeze(0)
        return pred.detach().cpu().to(dtype=self.dtype)

    def save(self, path_or_buffer):
        if self.model is None:
            raise ValueError("model is not fitted")
        payload = {
            "state_dict": self.model.state_dict(),
            "config": self.config,
            "trainii": self.trainii,
        }
        torch.save(payload, path_or_buffer)

    def load(self, path_or_buffer):
        payload = torch.load(path_or_buffer, map_location="cpu")
        self.trainii = payload["trainii"].to(torch.long)
        self.model = self._init_model(self.trainii)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        return self

'''


CONFIG_TEMPLATE = '''
<config>
  <constants
    cache_path="data/Cache"
    factor_root="data/Factor/FactorData"
    output_root="output"
    checkpoint_root=""
  />

  <strategy
    start_ds="20160111"
    end_ds="20200101"
  />

  <combo>
    <paths
      base_dir="."
      output_dir="output"
      model_path="Model.py"
      combo_base_path=""
      research_loader_path=""
      research_dataset_path=""
      checkpoint_root=""
    />

    <output
      alpha_history_path="output/alpha_history.pt"
      log_path="output/train.log"
      enable_alpha_analysis="true"
    />

    <runtime
      snaptime="mlp_minimal"
      livetrading="false"
      trainDelay="0"
      retDays="1"
      tsDays="8"
      load_chunk_days=""
      processed_feature_cache="false"
      torch_threads="64"
      torch_interop_threads="1"
      model_smooth_rate="0.7"
      model_keep_num="2"
      select_days="100"
      max_train_days="2000"
      verbose="false"
    />

    <model
      device="cpu"
      dropout="0.5"
      lr="0.000002"
      epochs="15"
      batch_size="3"
      weight_decay="0.000001"
      scheduler_step_size="10"
      scheduler_gamma="0.5"
      grad_clip="10.0"
      early_stopping_patience="5"
      seed="42"
    />

    <data
      dtype="float16"
      compression="none"
      data_start_ds="20160101"
      ashare_data_path=""
      valid_path=""
      filtered_path=""
      base_universe_path=""
    >
      <!-- Example factor resolves under constants.factor_root:
           data/Factor/FactorData/example_factor -->
      <item name="factor.example_factor" module="builtin.factor" path="example_factor" role="factor" display_name="example_factor" />

      <!-- Example label resolves under constants.cache_path:
           data/Cache/AshareCache/1d_DailyLabel/DailyLabel.vwap30_label1d -->
      <item name="label.example_label_1d" module="builtin.label" path="vwap30_label1d" role="label" />
    </data>

    <defaults
      selection_module=""
    />
  </combo>

  <backtest
    output_path="output/backtest"
    daily_metrics_file="daily_pnl.csv"
    cash="10000000.0"
    fee_rate="0.0015"
    reserve_cash="0.95"
    verbose="false"
    universe="base"
    execution_price="vwap30"
    drawdown_stop="0.0"
    cooldown_days="0"
  />

  <monitor
    enabled="false"
    output_path=""
    format="csv"
    print_summary="true"
    collect_gpu="true"
    sync_cuda="false"
    verbose="false"
  />
</config>
'''


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="combo-hello-world",
        description="Create a minimal editable comb2 experiment in the current directory.",
        epilog="example: combo-hello-world",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Create files without prompting for confirmation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Model.py, config.xml, and config.human",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target_dir = Path.cwd()
    targets = {
        "Model.py": _normalize_template(MODEL_TEMPLATE),
        "config.xml": _normalize_template(CONFIG_TEMPLATE),
        "config.human": load_config_human_template(),
    }

    existing = [name for name in targets if (target_dir / name).exists()]
    if existing and not args.force:
        joined = ", ".join(existing)
        print(f"combo-hello-world: refusing to overwrite existing file(s): {joined}", file=sys.stderr)
        print("combo-hello-world: rerun with --force to overwrite them", file=sys.stderr)
        return 2

    if not args.yes:
        print(f"combo-hello-world will create Model.py, config.xml, and config.human in: {target_dir}")
        answer = input("Continue? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("combo-hello-world: cancelled")
            return 1

    for name, content in targets.items():
        path = target_dir / name
        path.write_text(content, encoding="utf-8")
        print(f"created {path}")

    print("next: edit config.xml data paths, then run: runCombo config.xml")
    return 0


def _normalize_template(template: str) -> str:
    return textwrap.dedent(template).strip() + "\n"


def load_config_human_template() -> str:
    source_path = Path(__file__).resolve().with_name("config.human")
    if source_path.is_file():
        return source_path.read_text(encoding="utf-8")
    return resources.files("comb2_templates").joinpath("config.human").read_text(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
