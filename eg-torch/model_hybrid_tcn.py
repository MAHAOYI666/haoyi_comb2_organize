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


class HybridTCNModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        ts_days: int,
        hidden_size: int,
        fc_size: int,
        trainii: torch.Tensor,
        dropout: float = 0.5,
        tcn_channels: int = 128,
        tcn_layers: int = 1,
        tcn_kernel_size: int = 3,
        tcn_dropout: float | None = None,
        tcn_dilation_base: int = 1,
        use_tcn: bool = True,
        use_handcrafted: bool = True,
    ):
        super().__init__()
        if ts_days < 2:
            raise ValueError(f"HybridTCNModel requires ts_days >= 2, got {ts_days}")
        if not use_tcn and not use_handcrafted:
            raise ValueError("at least one of use_tcn/use_handcrafted must be true")
        if use_tcn:
            if tcn_layers < 1:
                raise ValueError(f"tcn_layers must be >= 1 when use_tcn=true, got {tcn_layers}")
            if tcn_channels < 1:
                raise ValueError(f"tcn_channels must be >= 1, got {tcn_channels}")
            if tcn_kernel_size < 1:
                raise ValueError(f"tcn_kernel_size must be >= 1, got {tcn_kernel_size}")
            if tcn_dilation_base < 1:
                raise ValueError(f"tcn_dilation_base must be >= 1, got {tcn_dilation_base}")

        self.trainii = trainii
        self.ts_days = ts_days
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_tcn = use_tcn
        self.use_handcrafted = use_handcrafted
        tcn_dropout = dropout if tcn_dropout is None else tcn_dropout

        self.dropout = nn.Dropout(dropout)
        if self.use_handcrafted:
            self.Q = nn.Linear(input_size, hidden_size)
            self.K = nn.Linear(input_size, hidden_size)
            self.hand_fc = nn.Linear(input_size * 6, hidden_size)

        if self.use_tcn:
            blocks: list[nn.Module] = []
            in_channels = input_size
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

        fused_input_size = hidden_size * int(self.use_handcrafted) + hidden_size * int(self.use_tcn)
        self.fuse_fc = nn.Linear(fused_input_size, hidden_size)
        self.mlpfc1 = nn.Linear(hidden_size, fc_size, bias=True)
        self.mlpfc2 = nn.Linear(fc_size, fc_size, bias=True)
        self.mlpfc3 = nn.Linear(fc_size, 1, bias=True)

    def attn(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, F] -> attention over the first T-1 days for each stock.
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

    def hand_branch(self, x: torch.Tensor) -> torch.Tensor:
        x_last = x[:, -1]  # [B, N, F]
        x_tsmean = x.mean(dim=1)  # [B, N, F]
        x_tsdelta = x[:, -1] - x[:, -2]  # [B, N, F]
        x_tsslope = self.ts_slope(x)  # [B, N, F]
        x_attn = self.attn(x)  # [B, N, F]
        hand_features = torch.cat(
            [x_last, x_tsmean, x_last - x_tsmean, x_tsdelta, x_tsslope, x_attn],
            dim=-1,
        )  # [B, N, 6F]
        out = self.hand_fc(hand_features)
        out = self.dropout(out)
        return F.leaky_relu(out, negative_slope=0.1)

    def tcn_branch(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_stocks, num_features = x.shape
        x_tcn = x.permute(0, 2, 3, 1).contiguous()
        x_tcn = x_tcn.view(batch_size * num_stocks, num_features, -1)  # [B*N, F, T]
        tcn_out = self.tcn(x_tcn)
        tcn_last = tcn_out[:, :, -1]  # [B*N, C]
        tcn_last = tcn_last.view(batch_size, num_stocks, -1)  # [B, N, C]
        out = self.tcn_fc(tcn_last)
        out = self.dropout(out)
        return F.leaky_relu(out, negative_slope=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, F]
        if x.dim() != 4:
            raise ValueError(f"expected x shape [B, T, N, F], got {tuple(x.shape)}")
        if x.shape[1] < 2:
            raise ValueError(f"HybridTCNModel requires T >= 2, got T={x.shape[1]}")

        x = x / 4.0
        branches = []
        if self.use_handcrafted:
            branches.append(self.hand_branch(x))
        if self.use_tcn:
            branches.append(self.tcn_branch(x))

        fused = torch.cat(branches, dim=-1)  # [B, N, hidden_size * num_branches]
        out = self.fuse_fc(fused)
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


Model = HybridTCNModel


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
        self.dtype = _as_dtype(self.config.get("dtype", torch.float16))
        self.ts_days = int(_cfg(self.config, "ts_days", "tsDays", 8))
        self.num_features = int(_cfg(self.config, "num_features", "numFeatures", 1))
        self.device = torch.device(self.config.get("device", "cpu"))
        self.hidden_size = int(_cfg(self.config, "hidden_size", "hiddenSize", 64))
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

        self.tcn_channels = int(_cfg(self.config, "tcn_channels", "tcnChannels", 128))
        self.tcn_layers = int(_cfg(self.config, "tcn_layers", "tcnLayers", 1))
        self.tcn_kernel_size = int(_cfg(self.config, "tcn_kernel_size", "tcnKernelSize", 3))
        self.tcn_dropout = float(_cfg(self.config, "tcn_dropout", "tcnDropout", self.dropout))
        self.tcn_dilation_base = int(_cfg(self.config, "tcn_dilation_base", "tcnDilationBase", 1))
        self.use_tcn = _as_bool(_cfg(self.config, "use_tcn", "useTcn", True))
        self.use_handcrafted = _as_bool(_cfg(self.config, "use_handcrafted", "useHandcrafted", True))

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
            tcn_channels=self.tcn_channels,
            tcn_layers=self.tcn_layers,
            tcn_kernel_size=self.tcn_kernel_size,
            tcn_dropout=self.tcn_dropout,
            tcn_dilation_base=self.tcn_dilation_base,
            use_tcn=self.use_tcn,
            use_handcrafted=self.use_handcrafted,
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
        self.config = payload.get("config", self.config)
        self._read_config()
        self._apply_seed()
        self.trainii = payload["trainii"].to(torch.long)
        self.model = self._init_model(self.trainii)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        return self
