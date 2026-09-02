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

from comb2 import FeatureGroups


class Model(nn.Module):
    def __init__(
        self,
        input_size: int,
        ts_days: int,
        hidden_size: int,
        fc_size: int,
        trainii: torch.Tensor,
        time_embedding_size: int | None,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.trainii = trainii
        self.dropout = nn.Dropout(dropout)
        self.Q = nn.Linear(input_size, hidden_size)
        self.K = nn.Linear(input_size, hidden_size)
        self.hdfc = nn.Linear(input_size * 6, hidden_size)
        self.time_embedding = (
            nn.Embedding(time_embedding_size, hidden_size)
            if time_embedding_size is not None
            else None
        )
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

    def forward(self, x: torch.Tensor, bar_id: torch.Tensor | None = None) -> torch.Tensor:
        x = x / 4.0
        x_last = x[:, -1]
        x_tsmean = x.mean(dim=1)
        x_tsdelta = x[:, -1] - x[:, -2]
        x_tsslope = self.ts_slope(x)
        x_attn = self.attn(x)
        x = torch.cat([x_last, x_tsmean, x_last - x_tsmean, x_tsdelta, x_tsslope, x_attn], dim=-1)

        out = self.hdfc(x)
        if self.time_embedding is not None:
            assert bar_id is not None
            out = out + self.time_embedding(bar_id).unsqueeze(1)
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
        weights = w.to(dtype=x.dtype)
        weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        x_centered = (x - (x * weights).sum(dim=1, keepdim=True) / weight_sum) * weights
        y_centered = (y - (y * weights).sum(dim=1, keepdim=True) / weight_sum) * weights
        numerator = torch.sum(x_centered * y_centered, dim=1)
        denominator = torch.sqrt(torch.sum(x_centered**2, dim=1) * torch.sum(y_centered**2, dim=1))
        correlation = torch.where(denominator > 1e-8, numerator / denominator.clamp_min(1e-8), 0.0)
        return (1 - correlation).mean()


TrainLoss = ICLoss


class ResearchModel:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.freq = str(config.get("freq", "1d"))
        self.dtype = config.get("dtype", torch.float16)
        self.ts_days = int(config.get("tsDays", 8))
        self.num_features = int(config.get("num_features_by_freq", {}).get("1d", config.get("num_features", 1)))
        self.target_times = tuple(int(value) for value in config.get("target_times", ()))
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
            time_embedding_size=None if self.freq == "1d" else len(self.target_times) + 1,
            dropout=self.dropout,
        )
        return model.to(self.device)

    def _next_batch(self, iterator):
        return next(iterator)

    def _batch_to_device(self, *batch):
        if self.freq == "1d":
            x, y, w = batch
            bar_id = None
        else:
            ti, x, y, w = batch
            bar_id = torch.as_tensor(
                [self.target_times.index(int(value)) + 1 for value in ti], dtype=torch.long
            ).to(self.device, non_blocking=True)
        return (
            bar_id,
            x.to(self.device, dtype=torch.float32, non_blocking=True),
            y.to(self.device, dtype=torch.float32, non_blocking=True),
            w.to(self.device, dtype=torch.float32, non_blocking=True),
        )

    def _zero_grad(self, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def _forward_batch(self, x: FeatureGroups, bar_id: torch.Tensor | None) -> torch.Tensor:
        return self.model(x["1d"], bar_id)

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
                    batch = self._next_batch(iterator)
                except StopIteration:
                    break
                if self.freq == "1d":
                    _, x, y, w = batch
                    bar_id, x, y, w = self._batch_to_device(x, y, w)
                else:
                    _, _, ti, x, y, w = batch
                    bar_id, x, y, w = self._batch_to_device(ti, x, y, w)
                self._zero_grad(optimizer)
                pred = self._forward_batch(x, bar_id)
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
    def predict(
        self, x_window: FeatureGroups, *, di: int | None = None, ti: int | None = None
    ) -> torch.Tensor:
        if self.model is None:
            raise ValueError("model is not fitted")
        x_1d = x_window["1d"]
        if x_1d.dim() != 3:
            raise ValueError(f"expected 3D 1d feature tensor, got shape={tuple(x_1d.shape)}")
        self.model.eval()
        x = x_1d.unsqueeze(0).to(self.device, dtype=torch.float32)
        if self.freq == "1d":
            bar_id = None
        else:
            assert di is not None and ti is not None
            bar_id = torch.tensor([self.target_times.index(int(ti)) + 1], device=self.device)
        pred = self.model(x, bar_id).squeeze(0)
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
    output_root="output"
    freq="1d"
  />

  <strategy
    start_ds="20160111"
    end_ds="20200101"
  >
    <optimizer
      type="opt1"
      lambda0="0.5"
      shrinkage="0.5"
      ret_days="60"
      ret_delay="1"
      ret_method="2"
      benchmark_delay="1"
      target_size="100000000.0"
      maxtvr="0.4"
      max_weight="0.0075"
      maxtrd="0.0"
      maxpos="0.0"
      liquidity_delay="1"
      lambda_slp="0.0"
      slippage_delay="1"
      min_participation_ratio="0.07"
      parti_penalty="0.0"
      trim_threshold="0.00001"
      min_valid_instruments="200"
      min_return_obs="20"
      soft_univ_penalty="0.00025"
      soft_risk_penalty="0.00004"
      soft_group_penalty="0.0002"
      num_mosek_threads="1"
      max_time="30.0"
      post_trim_renorm="false"
      univ_list="ZZ500:0.18:0.70,1|AshareST:0.00:0.00,1|AshareSH:0.00:0.60,1|AshareSZ:0.00:0.60,1|NONETOP3000:0.00:0.17,1|AshareCYB:0.10:0.30,1"
      soft_univ_list="ZZ1800:0.74:0.85:0.70,1|ZZ1800:0.79:0.85:0.45,1|ZZ1800:0.83:0.89:0.25,1|ZZ500:0.28:0.50:2.0,1|ZZ500:0.30:0.50:0.2,1|ZZ500:0.32:0.50:0.05,1|AshareSH:0.00:0.55:0.50,1|AshareCYB:0.10:0.25:1.00,1|AshareSZ:0.00:0.55:0.50,1|HS300:0.10:0.30:1.00,1|NONETOP3000:0.00:0.15:1.00,1"
      risk_list="cap:-0.40:0.30,1,4|cap:-0.20:0.21,1,2|returns120:-0.14:0.14,1|vola_30:-0.30:0.30,1|vola_5:-0.30:0.30,1|close:-0.10:0.10,1|BarraCNE5.BETA:-0.20:0.30,1|BarraCNE5.GROWTH:-0.15:0.20,1|BarraCNE5.BTOP:-0.15:0.20,1|BarraCNE5.LEVERAGE:-0.30:0.30,1|BarraCNE5.RESVOL:-0.30:0.30,1"
      soft_risk_list="cap:-0.02:0.07:2.0,1,4|returns120:-0.08:0.08:5.0,1|close:-0.05:0.05:1.0,1|BarraCNE5.BETA:0.00:0.04:1.7,1|BarraCNE5.GROWTH:-0.02:0.06:1.3,1|BarraCNE5.BTOP:-0.02:0.07:1.3,1|BarraCNE5.EARNYILD:-0.03:0.03:2.0,1"
      group_list="WindIndustry.sw1:-0.065:0.065,1"
      soft_group_list="WindIndustry.sw1:-0.05:0.05,1|WindIndustry.sw3:-0.012:0.012,1"
    />
  </strategy>

  <combo>
    <paths
      model_path="Model.py"
      combo_base_path=""
      research_loader_path=""
      research_dataset_path=""
    />

    <output
      enable_alpha_analysis="true"
    />

    <runtime
      snaptime="mlp_minimal"
      snap_ti=""
      livetrading="false"
      trainDelay="0"
      retDays="1"
      tsDays="8"
      load_chunk_days=""
      torch_threads="64"
      torch_interop_threads="1"
      model_smooth_rate="0.7"
      model_keep_num="2"
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
    />

    <data
      dtype="float16"
      compression="none"
      data_start_ds="20160101"
    >
      <!-- Factor paths can be absolute or relative to this config.xml. -->
      <item name="factor.example_factor" path="example_factor" role="factor" display_name="example_factor" />

      <!-- Daily labels resolve under constants.cache_path/AshareCache/1d_DailyLabel. -->
      <item name="label.example_label_1d" path="vwap30_label1d" role="label" />
    </data>

  </combo>

  <backtest
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
