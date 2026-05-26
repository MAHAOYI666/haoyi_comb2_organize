from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import torch


def turnover_penalty_objective(pi_array: np.ndarray, eta: float = 0.1, long_pos_weight: float = 1.0, long_pos_threshold: float = 0.0):
    def custom_loss(preds: np.ndarray, dataset: lgb.Dataset):
        y_true = dataset.get_label()
        pos_mask = y_true > long_pos_threshold
        weights = 1.0 + (long_pos_weight - 1.0) * pos_mask.astype(np.float32)
        grad = (preds - y_true) * weights
        hess = np.ones_like(grad) * weights

        valid_idx = np.where(pi_array >= 0)[0]
        if valid_idx.size:
            pi_valid = pi_array[valid_idx]
            turnover_diff = preds[valid_idx] - preds[pi_valid]
            grad[valid_idx] += 2 * eta * turnover_diff
            hess[valid_idx] += 2 * eta
        return grad, hess

    return custom_loss


class ResearchModel:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.dtype = config.get("dtype", torch.float16)
        self.ts_days = int(config.get("tsDays", 1))
        self.learning_rate = float(config.get("learning_rate", config.get("lr", 0.002)))
        self.num_boost_round = int(config.get("num_boost_round", config.get("epochs", 8000)))
        self.train_valid_split = float(config.get("train_valid_split", 0.85))
        self.earlystop = bool(config.get("earlystop", True))
        self.patience = int(config.get("patience", 100))
        self.reduce_turnover = bool(config.get("reduce_turnover", True))
        self.reduce_turnover_eta = float(config.get("reduce_turnover_eta", 0.1))
        self.seed = int(config.get("seed", 42))
        self.trainii: torch.Tensor | None = None
        self.model: lgb.Booster | None = None

    def _flatten_feature(self, x: torch.Tensor) -> np.ndarray:
        if x.dim() == 4:
            x = x[-1]
        if x.dim() != 3:
            raise ValueError(f"expected 3D or 4D feature tensor, got shape={tuple(x.shape)}")
        x = x.permute(1, 0, 2).contiguous()
        inst_count, ts_days, feat_dim = x.shape
        return x.reshape(inst_count, ts_days * feat_dim).to(torch.float32).cpu().numpy()

    def _collect_day_arrays(self, dataset, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        _, x, y, w = dataset[idx]
        x_flat = self._flatten_feature(x)
        y_np = y.to(torch.float32).cpu().numpy().reshape(-1)
        w_np = w.to(torch.float32).cpu().numpy().reshape(-1)
        valid_mask = np.isfinite(y_np) & np.isfinite(w_np) & (w_np > 0)
        sample_inst = np.arange(len(y_np), dtype=np.int32)[valid_mask]
        return x_flat[valid_mask], y_np[valid_mask], w_np[valid_mask], sample_inst

    def _collect_training_arrays(self, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        features = []
        labels = []
        weights = []
        groups = []
        sample_keys: list[tuple[int, int]] = []

        for idx in range(len(dataset)):
            x_flat, y_np, w_np, sample_inst = self._collect_day_arrays(dataset, idx)
            if y_np.size == 0:
                continue
            features.append(x_flat)
            labels.append(y_np)
            weights.append(w_np)
            groups.append(len(y_np))
            sample_keys.extend((idx, int(inst)) for inst in sample_inst)

        if not features:
            raise ValueError("no valid training samples available for LightGBM")

        key_to_pos = {key: pos for pos, key in enumerate(sample_keys)}
        pi_array = np.full(len(sample_keys), -1, dtype=np.int32)
        for pos, (day_idx, inst_idx) in enumerate(sample_keys):
            pi_array[pos] = key_to_pos.get((day_idx + 1, inst_idx), -1)

        return (
            np.concatenate(features, axis=0),
            np.concatenate(labels, axis=0),
            np.concatenate(weights, axis=0),
            np.asarray(groups, dtype=np.int32),
            pi_array,
        )

    def _split_train_valid(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, groups: np.ndarray, pi_array: np.ndarray):
        split_group = int(len(groups) * self.train_valid_split)
        split_group = min(max(split_group, 1), len(groups) - 1) if len(groups) > 1 else len(groups)
        split_row = int(groups[:split_group].sum())

        if split_row <= 0 or split_row >= len(y):
            train_group = groups
            valid_group = groups
            return x, y, w, train_group, pi_array, x, y, w, valid_group

        train_group = groups[:split_group]
        valid_group = groups[split_group:]
        train_pi = pi_array[:split_row].copy()
        train_pi[train_pi >= split_row] = -1
        return (
            x[:split_row],
            y[:split_row],
            w[:split_row],
            train_group,
            train_pi,
            x[split_row:],
            y[split_row:],
            w[split_row:],
            valid_group,
        )

    def fit(self, dataset):
        self.trainii = dataset.validinsts.detach().cpu().clone()
        x, y, w, groups, pi_array = self._collect_training_arrays(dataset)
        x_train, y_train, w_train, train_group, train_pi, x_valid, y_valid, w_valid, valid_group = self._split_train_valid(
            x, y, w, groups, pi_array
        )

        train_set = lgb.Dataset(x_train, label=y_train, weight=w_train, free_raw_data=False)
        valid_set = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, reference=train_set, free_raw_data=False)
        train_set.set_group(train_group)
        valid_set.set_group(valid_group)

        params = {
            "num_leaves": int(self.config.get("num_leaves", 127)),
            "max_depth": int(self.config.get("max_depth", 7)),
            "feature_fraction": float(self.config.get("feature_fraction", 0.8)),
            "bagging_fraction": float(self.config.get("bagging_fraction", 0.85)),
            "bagging_freq": int(self.config.get("bagging_freq", 2)),
            "verbose": int(self.config.get("verbose", -1)),
            "metric": self.config.get("metric", "mse"),
            "learning_rate": self.learning_rate,
            "num_threads": int(self.config.get("num_threads", -1)),
            "seed": self.seed,
        }
        if not self.reduce_turnover:
            params["objective"] = self.config.get("objective", "regression")
        else:
            params["objective"] = turnover_penalty_objective(train_pi, eta=self.reduce_turnover_eta)

        callbacks = []
        if self.earlystop and len(valid_group) > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=self.patience, verbose=True))

        self.model = lgb.train(
            params=params,
            train_set=train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=[train_set, valid_set],
            valid_names=["Train", "Valid"],
            callbacks=callbacks,
        )
        print(
            f"[FIT] train_samples={x_train.shape[0]} valid_samples={x_valid.shape[0]} "
            f"features={x_train.shape[1]} num_boost_round={self.num_boost_round}"
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
