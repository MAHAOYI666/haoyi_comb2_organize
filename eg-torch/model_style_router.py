from __future__ import annotations

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


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, C, T], pad only on the left to keep the convolution causal.
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.conv(x)
        out = F.leaky_relu(out, negative_slope=0.1)
        out = self.dropout(out)
        out = out + residual
        return F.leaky_relu(out, negative_slope=0.1)


class ExpertHead(nn.Module):
    def __init__(self, input_size: int, fc_size: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_size, fc_size, bias=True)
        self.fc2 = nn.Linear(fc_size, fc_size, bias=True)
        self.fc3 = nn.Linear(fc_size, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.dropout(out)
        out = F.leaky_relu(out, negative_slope=0.1)
        out = self.fc2(out)
        out = self.dropout(out)
        out = F.leaky_relu(out, negative_slope=0.1)
        out = self.fc3(out)
        return out.squeeze(-1)


class Model(nn.Module):
    def __init__(
        self,
        input_size: int,
        ts_days: int,
        hidden_size: int,
        fc_size: int,
        trainii: torch.Tensor,
        dropout: float = 0.5,
        alpha_feature_count: int | None = None,
        style_feature_count: int = 0,
        tcn_channels: int = 128,
        tcn_kernel_size: int = 3,
        gate_hidden_size: int = 64,
        gate_dropout: float = 0.1,
        tcn_layers: int = 1,
        tcn_dropout: float | None = None,
        tcn_dilation_base: int = 1,
    ):
        super().__init__()
        self.trainii = trainii
        self.ts_days = int(ts_days)
        self.input_size = int(input_size)
        self.alpha_feature_count = int(input_size if alpha_feature_count is None else alpha_feature_count)
        self.style_feature_count = int(style_feature_count)
        self.total_feature_count = self.alpha_feature_count + self.style_feature_count
        self.last_gate_weights: torch.Tensor | None = None

        if self.alpha_feature_count < 1:
            raise ValueError(f"alpha_feature_count must be >= 1, got {self.alpha_feature_count}")
        if self.style_feature_count < 0:
            raise ValueError(f"style_feature_count must be >= 0, got {self.style_feature_count}")
        if self.input_size != self.total_feature_count:
            raise ValueError(
                "input_size must equal alpha_feature_count + style_feature_count, "
                f"got input_size={self.input_size}, alpha_feature_count={self.alpha_feature_count}, "
                f"style_feature_count={self.style_feature_count}, expected={self.total_feature_count}"
            )
        if tcn_layers < 1:
            raise ValueError(f"tcn_layers must be >= 1, got {tcn_layers}")
        if tcn_channels < 1:
            raise ValueError(f"tcn_channels must be >= 1, got {tcn_channels}")
        if tcn_kernel_size < 1:
            raise ValueError(f"tcn_kernel_size must be >= 1, got {tcn_kernel_size}")
        if tcn_dilation_base < 1:
            raise ValueError(f"tcn_dilation_base must be >= 1, got {tcn_dilation_base}")

        tcn_dropout = dropout if tcn_dropout is None else tcn_dropout
        self.dropout = nn.Dropout(dropout)

        self.Q = nn.Linear(self.alpha_feature_count, hidden_size)
        self.K = nn.Linear(self.alpha_feature_count, hidden_size)
        self.hand_fc = nn.Linear(self.alpha_feature_count * 6, hidden_size)
        self.hand_head = ExpertHead(hidden_size, fc_size, dropout)

        blocks: list[nn.Module] = []
        in_channels = self.alpha_feature_count
        for layer_idx in range(tcn_layers):
            dilation = tcn_dilation_base**layer_idx
            blocks.append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=tcn_channels,
                    kernel_size=tcn_kernel_size,
                    dilation=dilation,
                    dropout=tcn_dropout,
                )
            )
            in_channels = tcn_channels
        self.tcn = nn.Sequential(*blocks)
        self.tcn_fc = nn.Linear(tcn_channels, hidden_size)
        self.tcn_head = ExpertHead(hidden_size, fc_size, dropout)

        self.global_gate_logits = nn.Parameter(torch.zeros(2))
        if self.style_feature_count > 0:
            self.gate_net = nn.Sequential(
                nn.Linear(self.style_feature_count, gate_hidden_size),
                nn.Dropout(gate_dropout),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(gate_hidden_size, 2),
            )
        else:
            self.gate_net = None

    def _split_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"expected x shape [B, T, N, F], got {tuple(x.shape)}")
        if x.shape[1] < 1:
            raise ValueError(f"expected at least one time step, got T={x.shape[1]}")
        got = int(x.shape[-1])
        expected = self.alpha_feature_count + self.style_feature_count
        if got != expected:
            raise ValueError(
                "feature dimension mismatch: "
                f"got={got}, expected={expected}, "
                f"alpha_feature_count={self.alpha_feature_count}, "
                f"style_feature_count={self.style_feature_count}"
            )
        x_alpha = x[..., : self.alpha_feature_count]
        x_style = x[..., self.alpha_feature_count : self.alpha_feature_count + self.style_feature_count]
        return x_alpha, x_style

    def attn(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, F], attention over the first T-1 days for each stock.
        if x.shape[1] < 2:
            return torch.zeros_like(x[:, -1])
        x = x.permute(0, 2, 1, 3).contiguous()
        q = self.Q(x[:, :, -1])
        k = self.K(x[:, :, :-1])
        attn_scores = torch.matmul(q.unsqueeze(2), k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, x[:, :, :-1]).squeeze(2)

    def ts_slope(self, x: torch.Tensor) -> torch.Tensor:
        _, steps, _, _ = x.shape
        if steps < 2:
            return torch.zeros_like(x[:, -1])
        ts = torch.arange(steps, dtype=x.dtype, device=x.device)
        ts = ts - ts.mean()
        ts_var = ts.pow(2).sum()
        cov = (ts.view(1, steps, 1, 1) * (x - x.mean(dim=1, keepdim=True))).sum(dim=1)
        return cov / ts_var

    def hand_branch(self, x_alpha: torch.Tensor) -> torch.Tensor:
        x_last = x_alpha[:, -1]
        x_tsmean = x_alpha.mean(dim=1)
        if x_alpha.shape[1] >= 2:
            x_tsdelta = x_alpha[:, -1] - x_alpha[:, -2]
        else:
            x_tsdelta = torch.zeros_like(x_last)
        x_tsslope = self.ts_slope(x_alpha)
        x_attn = self.attn(x_alpha)
        hand_features = torch.cat(
            [x_last, x_tsmean, x_last - x_tsmean, x_tsdelta, x_tsslope, x_attn],
            dim=-1,
        )
        out = self.hand_fc(hand_features)
        out = self.dropout(out)
        return F.leaky_relu(out, negative_slope=0.1)

    def tcn_branch(self, x_alpha: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_stocks, num_features = x_alpha.shape
        x_tcn = x_alpha.permute(0, 2, 3, 1).contiguous()
        x_tcn = x_tcn.view(batch_size * num_stocks, num_features, -1)
        tcn_out = self.tcn(x_tcn)
        tcn_last = tcn_out[:, :, -1]
        tcn_last = tcn_last.view(batch_size, num_stocks, -1)
        out = self.tcn_fc(tcn_last)
        out = self.dropout(out)
        return F.leaky_relu(out, negative_slope=0.1)

    def gate_branch(self, x_style: torch.Tensor, batch_size: int, num_stocks: int) -> torch.Tensor:
        if self.style_feature_count > 0:
            style_last = x_style[:, -1]
            gate_logits = self.gate_net(style_last)
        else:
            gate_logits = self.global_gate_logits.view(1, 1, 2).expand(batch_size, num_stocks, 2)
        gate_weights = F.softmax(gate_logits, dim=-1)
        self.last_gate_weights = gate_weights.detach()
        return gate_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_alpha, x_style = self._split_features(x)
        x_alpha = x_alpha / 4.0
        if self.style_feature_count > 0:
            x_style = x_style / 4.0

        hand_repr = self.hand_branch(x_alpha)
        tcn_repr = self.tcn_branch(x_alpha)
        alpha_hand = self.hand_head(hand_repr)
        alpha_tcn = self.tcn_head(tcn_repr)

        gate_weights = self.gate_branch(x_style, batch_size=x.shape[0], num_stocks=x.shape[2])
        final_alpha = gate_weights[..., 0] * alpha_hand + gate_weights[..., 1] * alpha_tcn
        return final_alpha


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
        self._read_config()
        self.trainii: torch.Tensor | None = None
        self.model: Model | None = None
        self.loss_fn = TrainLoss()
        self._apply_seed()

    def _read_config(self) -> None:
        self.architecture = str(self.config.get("architecture", "hand_tcn_style_router"))
        self.dtype = _as_dtype(self.config.get("dtype", torch.float16))
        self.ts_days = int(_cfg(self.config, "ts_days", "tsDays", 8))
        self.num_features = int(_cfg(self.config, "num_features", "numFeatures", 1))
        self.device = torch.device(self.config.get("device", "cpu"))
        self.hidden_size = int(_cfg(self.config, "hidden_size", "hiddenSize", 512))
        self.fc_size = int(_cfg(self.config, "fc_size", "fcSize", self.hidden_size))
        self.dropout = float(self.config.get("dropout", 0.5))
        self.lr = float(self.config.get("lr", 2e-6))
        self.epochs = int(self.config.get("epochs", 15))
        self.batch_size = int(_cfg(self.config, "batch_size", "batchSize", 3))
        self.weight_decay = float(self.config.get("weight_decay", 1e-6))
        self.scheduler_step_size = int(self.config.get("scheduler_step_size", 10))
        self.scheduler_gamma = float(self.config.get("scheduler_gamma", 0.5))
        self.grad_clip = float(self.config.get("grad_clip", 10.0))
        self.early_stopping_patience = int(self.config.get("early_stopping_patience", 5))
        self.seed = int(self.config.get("seed", 42))

        self.style_feature_count = int(_cfg(self.config, "style_feature_count", "styleFeatureCount", 0) or 0)
        alpha_feature_count = _cfg(self.config, "alpha_feature_count", "alphaFeatureCount", None)
        if alpha_feature_count is None:
            if self.style_feature_count == 0:
                self.alpha_feature_count = self.num_features
            else:
                raise ValueError(
                    "alpha_feature_count is required when style_feature_count > 0; "
                    f"got style_feature_count={self.style_feature_count}"
                )
        else:
            self.alpha_feature_count = int(alpha_feature_count)

        expected_total = self.alpha_feature_count + self.style_feature_count
        total_feature_count = _cfg(self.config, "total_feature_count", "totalFeatureCount", None)
        self.total_feature_count = expected_total if total_feature_count is None else int(total_feature_count)
        if self.total_feature_count != expected_total:
            raise ValueError(
                "total_feature_count must equal alpha_feature_count + style_feature_count, "
                f"got total_feature_count={self.total_feature_count}, expected={expected_total}"
            )
        if self.num_features != self.total_feature_count:
            raise ValueError(
                "num_features must match total_feature_count, "
                f"got num_features={self.num_features}, total_feature_count={self.total_feature_count}"
            )

        configured_use_style_gate = _as_bool(
            _cfg(self.config, "use_style_gate", "useStyleGate", self.style_feature_count > 0)
        )
        self.configured_use_style_gate = configured_use_style_gate
        self.use_style_gate = self.style_feature_count > 0
        self.tcn_channels = int(_cfg(self.config, "tcn_channels", "tcnChannels", 128))
        self.tcn_layers = int(_cfg(self.config, "tcn_layers", "tcnLayers", 1))
        self.tcn_kernel_size = int(_cfg(self.config, "tcn_kernel_size", "tcnKernelSize", 3))
        self.tcn_dropout = float(_cfg(self.config, "tcn_dropout", "tcnDropout", self.dropout))
        self.tcn_dilation_base = int(_cfg(self.config, "tcn_dilation_base", "tcnDilationBase", 1))
        self.gate_hidden_size = int(_cfg(self.config, "gate_hidden_size", "gateHiddenSize", 64))
        self.gate_dropout = float(_cfg(self.config, "gate_dropout", "gateDropout", 0.1))
        self.log_gate_stats = _as_bool(_cfg(self.config, "log_gate_stats", "logGateStats", True))

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
            input_size=self.total_feature_count,
            ts_days=self.ts_days,
            hidden_size=self.hidden_size,
            fc_size=self.fc_size,
            trainii=trainii,
            dropout=self.dropout,
            alpha_feature_count=self.alpha_feature_count,
            style_feature_count=self.style_feature_count,
            tcn_channels=self.tcn_channels,
            tcn_kernel_size=self.tcn_kernel_size,
            gate_hidden_size=self.gate_hidden_size,
            gate_dropout=self.gate_dropout,
            tcn_layers=self.tcn_layers,
            tcn_dropout=self.tcn_dropout,
            tcn_dilation_base=self.tcn_dilation_base,
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

    def _log_model_config(self) -> None:
        print(f"[MODEL] architecture={self.architecture}")
        print(f"[MODEL] alpha_feature_count={self.alpha_feature_count}")
        print(f"[MODEL] style_feature_count={self.style_feature_count}")
        print(f"[MODEL] total_feature_count={self.total_feature_count}")
        print(f"[MODEL] use_style_gate={self.use_style_gate}")

    def _batch_gate_stats(self) -> dict[str, float] | None:
        if not self.log_gate_stats or self.model is None:
            return None
        gate_weights = getattr(self.model, "last_gate_weights", None)
        if gate_weights is None:
            return None
        gate = gate_weights.detach().to(dtype=torch.float32)
        return {
            "gate_hand_mean": float(gate[..., 0].mean().cpu()),
            "gate_tcn_mean": float(gate[..., 1].mean().cpu()),
            "gate_hand_std": float(gate[..., 0].std(unbiased=False).cpu()),
            "gate_tcn_std": float(gate[..., 1].std(unbiased=False).cpu()),
        }

    @staticmethod
    def _accumulate_gate_stats(accumulator: dict[str, float], stats: dict[str, float] | None) -> None:
        if stats is None:
            return
        accumulator["count"] = accumulator.get("count", 0.0) + 1.0
        for key, value in stats.items():
            accumulator[key] = accumulator.get(key, 0.0) + value

    def _format_gate_stats(self, accumulator: dict[str, float]) -> str:
        count = accumulator.get("count", 0.0)
        if not self.log_gate_stats or count <= 0:
            return ""
        return (
            f" gate_hand_mean={accumulator['gate_hand_mean'] / count:.4f}"
            f" gate_tcn_mean={accumulator['gate_tcn_mean'] / count:.4f}"
            f" gate_hand_std={accumulator['gate_hand_std'] / count:.4f}"
            f" gate_tcn_std={accumulator['gate_tcn_std'] / count:.4f}"
        )

    def fit(self, dataset):
        self._apply_seed()
        self._log_model_config()
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
            gate_stats: dict[str, float] = {}
            iterator = iter(dataloader)
            while True:
                try:
                    _, x, y, w = self._next_batch(iterator)
                except StopIteration:
                    break
                x, y, w = self._batch_to_device(x, y, w)
                self._zero_grad(optimizer)
                pred = self._forward_batch(x)
                self._accumulate_gate_stats(gate_stats, self._batch_gate_stats())
                loss = self._compute_loss(pred, y, w)
                self._backward_loss(loss)
                self._clip_grad()
                self._optimizer_step(optimizer)
                total_loss += self._loss_to_float(loss)
                batches += 1
            scheduler.step()
            avg_loss = total_loss / max(batches, 1)
            print(
                f"[FIT] epoch={epoch + 1}/{self.epochs} loss={avg_loss:.6f}"
                f"{self._format_gate_stats(gate_stats)}"
            )
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
        self.config = payload.get("config", self.config)
        self._read_config()
        self._apply_seed()
        self.trainii = payload["trainii"].to(torch.long)
        self.model = self._init_model(self.trainii)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        return self
