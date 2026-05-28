from __future__ import annotations

# tcn_only ablation: the only architectural change is removing the TCN stack and
# feeding the last-day factors directly into the MLP head, to test whether the
# original TCN branch learned useful temporal structure.

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


def _cfg(config: dict[str, Any], snake_name: str, camel_name: str, default: Any) -> Any:
    return config.get(snake_name, config.get(camel_name, default))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _as_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in DTYPE_MAP:
            return DTYPE_MAP[key]
    raise ValueError(f"unsupported dtype: {value}")


def _init_output_head(linear: nn.Linear) -> None:
    nn.init.normal_(linear.weight, mean=0.0, std=0.02)
    nn.init.normal_(linear.bias, mean=0.0, std=0.1)


def _torch_load(path_or_buffer):
    try:
        return torch.load(path_or_buffer, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path_or_buffer, map_location="cpu")


class MultiHeadICLoss(nn.Module):
    def __init__(self, num_heads: int, decorr_lam: float = 0.05):
        super().__init__()
        self.H = int(num_heads)
        self.decorr_lam = float(decorr_lam)
        self.last_ic_mean: torch.Tensor | None = None
        self.last_decorr: torch.Tensor | None = None

    def _ic_per_head(self, x, y, w):
        x = x * w
        y = y * w
        y = y - y.mean(dim=1, keepdim=True)
        x = x - x.mean(dim=1, keepdim=True)
        x = x * w
        y = y * w
        num = (x * y).sum(dim=1)
        den = torch.sqrt((x * x).sum(dim=1) + 1e-8)
        return (1 - num / den).mean()

    def forward(self, pred, y, w):
        ics = torch.stack([self._ic_per_head(pred[..., h], y, w) for h in range(self.H)])
        ic_mean = ics.mean()
        self.last_ic_mean = ic_mean.detach()
        if self.H == 1 or self.decorr_lam == 0.0:
            self.last_decorr = torch.zeros((), device=pred.device)
            return ic_mean

        w_e = w.unsqueeze(-1)
        p = pred * w_e
        p = p - p.mean(dim=1, keepdim=True)
        p = p * w_e
        cov = torch.einsum("bnh,bnk->bhk", p, p)
        std = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2) + 1e-8)
        corr = cov / (std.unsqueeze(-1) * std.unsqueeze(-2) + 1e-8)
        mask = torch.triu(torch.ones(self.H, self.H, device=pred.device, dtype=torch.bool), diagonal=1)
        decorr = corr.abs()[:, mask].mean()
        self.last_decorr = decorr.detach()
        return ic_mean + self.decorr_lam * decorr


class LastDayMLPModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        ts_days: int,
        hidden_size: int,
        fc_size: int,
        trainii: torch.Tensor,
        dropout: float = 0.5,
        num_heads: int = 4,
    ):
        super().__init__()
        if ts_days < 1:
            raise ValueError(f"LastDayMLPModel requires ts_days >= 1, got {ts_days}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")

        self.trainii = trainii
        self.ts_days = ts_days
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.fuse_fc = nn.Linear(hidden_size, hidden_size)
        self.mlpfc1 = nn.Linear(hidden_size, fc_size, bias=True)
        self.mlpfc2 = nn.Linear(fc_size, fc_size, bias=True)
        self.mlpfc3 = nn.Linear(fc_size, num_heads, bias=True)
        _init_output_head(self.mlpfc3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected x shape [B, T, N, F], got {tuple(x.shape)}")

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x / 4.0
        x_last = x[:, -1]
        out = self.input_proj(x_last)
        out = self.fuse_fc(out)
        out = self.dropout(out)
        out = F.leaky_relu(out, negative_slope=0.1)
        out = self.mlpfc1(out)
        out = self.dropout(out)
        out = F.leaky_relu(out, negative_slope=0.1)
        out = self.mlpfc2(out)
        out = self.dropout(out)
        out = F.leaky_relu(out, negative_slope=0.1)
        return self.mlpfc3(out)


Model = LastDayMLPModel
TrainLoss = MultiHeadICLoss


class ResearchModel:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._read_config()
        self.trainii: torch.Tensor | None = None
        self.model: Model | None = None
        self.loss_fn = TrainLoss(num_heads=self.num_heads, decorr_lam=self.decorr_lam)
        self._apply_seed()

    def _read_config(self) -> None:
        self.dtype = _as_dtype(self.config.get("dtype", torch.float16))
        self.ts_days = int(_cfg(self.config, "ts_days", "tsDays", 5))
        self.num_features = int(_cfg(self.config, "num_features", "numFeatures", 1))
        self.device = torch.device(self.config.get("device", "cpu"))
        self.hidden_size = int(_cfg(self.config, "hidden_size", "hiddenSize", 512))
        self.fc_size = int(_cfg(self.config, "fc_size", "fcSize", 256))
        self.dropout = float(self.config.get("dropout", 0.5))
        self.lr = float(self.config.get("lr", 2e-6))
        self.epochs = int(self.config.get("epochs", 20))
        self.batch_size = int(_cfg(self.config, "batch_size", "batchSize", 3))
        self.weight_decay = float(self.config.get("weight_decay", 1e-6))
        self.scheduler_step_size = int(self.config.get("scheduler_step_size", 10))
        self.scheduler_gamma = float(self.config.get("scheduler_gamma", 0.5))
        self.grad_clip = float(self.config.get("grad_clip", 10.0))
        self.early_stopping_patience = int(_cfg(self.config, "early_stopping_patience", "earlyStoppingPatience", 5))
        self.seed = int(self.config.get("seed", 42))

        self.num_heads = int(_cfg(self.config, "num_heads", "numHeads", 4))
        self.decorr_lam = float(_cfg(self.config, "decorr_lam", "decorrLam", 0.05))

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
            num_heads=self.num_heads,
        )
        return model.to(self.device)

    def _next_batch(self, iterator):
        return next(iterator)

    def _batch_to_device(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
        x = x.to(self.device, dtype=torch.float32, non_blocking=True)
        y = y.to(self.device, dtype=torch.float32, non_blocking=True)
        w = w.to(self.device, dtype=torch.float32, non_blocking=True)
        return (
            torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0),
        )

    def _zero_grad(self, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def _forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_loss(self, pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(pred, y, w)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: {float(loss.detach().cpu())}")
        return loss

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
        best_ic = float("inf")
        best_state_dict = None
        stale_epochs = 0

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_ic = 0.0
            total_decorr = 0.0
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

                ic_value = self.loss_fn.last_ic_mean
                decorr_value = self.loss_fn.last_decorr
                total_loss += self._loss_to_float(loss)
                total_ic += self._loss_to_float(ic_value)
                total_decorr += self._loss_to_float(decorr_value)
                batches += 1
            scheduler.step()
            avg_loss = total_loss / max(batches, 1)
            avg_ic = total_ic / max(batches, 1)
            avg_decorr = total_decorr / max(batches, 1)
            print(
                f"[FIT] epoch={epoch + 1}/{self.epochs} "
                f"loss={avg_loss:.6f} ic_mean={avg_ic:.6f} decorr={avg_decorr:.6f}"
            )
            if avg_ic < best_ic:
                best_ic = avg_ic
                best_state_dict = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.early_stopping_patience:
                    print(
                        f"[FIT] early stopping at epoch={epoch + 1}/{self.epochs} "
                        f"best_ic_mean={best_ic:.6f}"
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
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        raw = self.model(x).squeeze(0)
        pred = raw.mean(dim=-1)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
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
        payload = _torch_load(path_or_buffer)
        self.config = payload.get("config", self.config)
        self._read_config()
        self.loss_fn = TrainLoss(num_heads=self.num_heads, decorr_lam=self.decorr_lam)
        self._apply_seed()
        self.trainii = payload["trainii"].to(torch.long)
        self.model = self._init_model(self.trainii)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        return self
