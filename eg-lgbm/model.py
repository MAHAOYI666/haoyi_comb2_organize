from __future__ import annotations

from io import BufferedIOBase, BytesIO
from typing import Any

import lightgbm as lgb
import numpy as np
import torch


class ResearchModel:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.dtype = config.get("dtype", torch.float16)
        self.ts_days = int(config.get("tsDays", 16))
        self.num_features = int(config.get("num_features_by_freq", {}).get("1d", config.get("num_features", 1)))
        self.learning_rate = float(config.get("lr", 0.05))
        self.num_boost_round = int(config.get("epochs", 100))
        self.trainii: torch.Tensor | None = None
        self.model: lgb.Booster | None = None

    def _flatten_feature(self, x: torch.Tensor) -> np.ndarray:
        x = x["1d"]
        if x.dim() == 4:
            x = x[-1]
        if x.dim() != 3:
            raise ValueError(f"expected 3D or 4D feature tensor, got shape={tuple(x.shape)}")
        x = x.permute(1, 0, 2).contiguous()
        inst_count, ts_days, feat_dim = x.shape
        return x.reshape(inst_count, ts_days * feat_dim).to(torch.float32).cpu().numpy()

    def _collect_training_arrays(self, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = []
        labels = []
        weights = []
        for idx in range(len(dataset)):
            _, x, y, w = dataset[idx]
            x_flat = self._flatten_feature(x)
            y_np = y.to(torch.float32).cpu().numpy()
            w_np = w.to(torch.float32).cpu().numpy()
            valid_mask = np.isfinite(y_np) & np.isfinite(w_np) & (w_np > 0)
            if not np.any(valid_mask):
                continue
            features.append(x_flat[valid_mask])
            labels.append(y_np[valid_mask])
            weights.append(w_np[valid_mask])
        if not features:
            raise ValueError("no valid training samples available for LightGBM")
        return (
            np.concatenate(features, axis=0),
            np.concatenate(labels, axis=0),
            np.concatenate(weights, axis=0),
        )

    def fit(self, dataset):
        self.trainii = dataset.validinsts.detach().cpu().clone()
        x_train, y_train, w_train = self._collect_training_arrays(dataset)
        train_set = lgb.Dataset(x_train, label=y_train, weight=w_train, free_raw_data=False)
        params = {
            "objective": "regression",
            "metric": "l2",
            "verbosity": -1,
            "learning_rate": self.learning_rate,
            "num_leaves": int(self.config.get("num_leaves", 31)),
            "feature_fraction": float(self.config.get("feature_fraction", 0.8)),
            "bagging_fraction": float(self.config.get("bagging_fraction", 0.8)),
            "bagging_freq": int(self.config.get("bagging_freq", 1)),
            "min_data_in_leaf": int(self.config.get("min_data_in_leaf", 100)),
            "num_threads": int(self.config.get("num_threads", -1)),
            "seed": int(self.config.get("seed", 42)),
        }
        self.model = lgb.train(params, train_set, num_boost_round=self.num_boost_round)
        print(
            f"[FIT] samples={x_train.shape[0]} features={x_train.shape[1]} "
            f"num_boost_round={self.num_boost_round}"
        )
        return self

    def predict(self, x_window: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ValueError("model is not fitted")
        x = self._flatten_feature(x_window)
        pred = self.model.predict(x)
        return torch.as_tensor(pred, dtype=self.dtype)

    def save(self, path_or_buffer):
        if self.model is None:
            raise ValueError("model is not fitted")
        payload = {
            "model_str": self.model.model_to_string(),
            "config": self.config,
            "trainii": self.trainii,
        }
        torch.save(payload, path_or_buffer)

    def load(self, path_or_buffer):
        payload = torch.load(path_or_buffer, map_location="cpu")
        self.model = lgb.Booster(model_str=payload["model_str"])
        self.trainii = payload.get("trainii")
        return self
