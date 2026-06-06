from __future__ import annotations

from typing import Any

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
        self.hidden_size = int(config.get("hidden_size", 64))
        self.fc_size = int(config.get("fc_size", self.hidden_size))
        self.dropout = float(config.get("dropout", 0.5))
        self.lr = float(config.get("lr", 2e-6))
        self.epochs = int(config.get("epochs", 15))
        self.batch_size = int(config.get("batch_size", 3))
        self.weight_decay = float(config.get("weight_decay", 1e-6))
        self.scheduler_step_size = int(config.get("scheduler_step_size", 10))
        self.scheduler_gamma = float(config.get("scheduler_gamma", 0.5))
        self.grad_clip = float(config.get("grad_clip", 10.0))
        self.trainii: torch.Tensor | None = None
        self.model: Model | None = None
        self.loss_fn = TrainLoss()

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
            x.to(self.device, dtype=torch.float32),
            y.to(self.device, dtype=torch.float32),
            w.to(self.device, dtype=torch.float32),
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
        self.trainii = dataset.validinsts.detach().cpu().to(torch.long).clone()
        self.model = self._init_model(self.trainii)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()

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
            print(f"[FIT] epoch={epoch + 1}/{self.epochs} loss={total_loss / max(batches, 1):.6f}")
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
