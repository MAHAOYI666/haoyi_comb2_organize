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
        self.num_features = int(config["num_features"])
        self.learning_rate = float(config.get("lr", 0.05))
        self.num_boost_round = int(config.get("epochs", 100))
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

    def _collect_training_arrays(self, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = []
        labels = []
        weights = []
        for idx in range(len(dataset)):
            _, ds, ti, x, y, w = dataset[idx]
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

    def predict(self, x_window: torch.Tensor, *, di: int, ti: int) -> torch.Tensor:
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


from pathlib import Path
from comb2 import ComboDataLoader, DataItem
from comb2_simbase.cache_layout import ashare_cache_path


class ResearchLoader(ComboDataLoader):
    """Same-time daily factors and the researcher-indexed snapshot label."""

    def data_requirements(self):
        root = ashare_cache_path(self.config.cache_path)
        return (
            DataItem("alpha.cyz_20231130_01", path="factors/cyz_20231130_01", delay=1),
            DataItem("alpha.cyz_20231130_02", path="factors/cyz_20231130_02", delay=1),
            DataItem("alpha.cyz_20231130_03", path="factors/cyz_20231130_03", delay=1),
            DataItem("alpha.cyz_20240718_01", path="factors/cyz_20240718_01", delay=1),
            DataItem("alpha.cyz_20240718_02", path="factors/cyz_20240718_02", delay=1),
            DataItem("alpha.cyz_20240212_02", path="factors/cyz_20240212_02", delay=1),
            DataItem("alpha.cyz_20240212_01", path="factors/cyz_20240212_01", delay=1),
            DataItem("alpha.cyz_20240208_01", path="factors/cyz_20240208_01", delay=1),
            DataItem("alpha.cyz_20231026_01", path="factors/cyz_20231026_01", delay=1),
            DataItem("alpha.cyz_20231026_02", path="factors/cyz_20231026_02", delay=1),
            DataItem("alpha.cyz_20240627_1", path="factors/cyz_20240627_1", delay=1),
            DataItem("alpha.cyz_20240627_2", path="factors/cyz_20240627_2", delay=1),
            DataItem("alpha.cyz_20240627_3", path="factors/cyz_20240627_3", delay=1),
            DataItem("alpha.cyz_20240627_4", path="factors/cyz_20240627_4", delay=1),
            DataItem("alpha.cyz_20231019_01", path="factors/cyz_20231019_01", delay=1),
            DataItem("alpha.cyz_20231019_02", path="factors/cyz_20231019_02", delay=1),
            DataItem("alpha.cyz_20231207_01", path="factors/cyz_20231207_01", delay=1),
            DataItem("alpha.cyz_20231207_02", path="factors/cyz_20231207_02", delay=1),
            DataItem("alpha.cyz_20231207_03", path="factors/cyz_20231207_03", delay=1),
            DataItem("alpha.cyz_20231207_04", path="factors/cyz_20231207_04", delay=1),
            DataItem("alpha.cyz_20231207_05", path="factors/cyz_20231207_05", delay=1),
            DataItem("alpha.cyz_20240425_2", path="factors/cyz_20240425_2", delay=1),
            DataItem("alpha.cyz_20240425_3", path="factors/cyz_20240425_3", delay=1),
            DataItem("alpha.cyz_20240425_5", path="factors/cyz_20240425_5", delay=1),
            DataItem("alpha.cyz_20240425_4", path="factors/cyz_20240425_4", delay=1),
            DataItem("alpha.cyz_20240425_6", path="factors/cyz_20240425_6", delay=1),
            DataItem("alpha.cyz_20231123_01", path="factors/cyz_20231123_01", delay=1),
            DataItem("alpha.cyz_20240219_01", path="factors/cyz_20240219_01", delay=1),
            DataItem("alpha.cyz_20240220_02", path="factors/cyz_20240220_02", delay=1),
            DataItem("alpha.cyz_20240222_03", path="factors/cyz_20240222_03", delay=1),
            DataItem("alpha.cyz_20240226_04", path="factors/cyz_20240226_04", delay=1),
            DataItem("alpha.wjx_20231130_01", path="factors/wjx_20231130_01", delay=1),
            DataItem("alpha.wjx_20231130_02", path="factors/wjx_20231130_02", delay=1),
            DataItem("alpha.wjx_20240523_01", path="factors/wjx_20240523_01", delay=1),
            DataItem("alpha.wjx_20240523_02", path="factors/wjx_20240523_02", delay=1),
            DataItem("alpha.wjx_20240523_03", path="factors/wjx_20240523_03", delay=1),
            DataItem("alpha.wjx_20240523_04", path="factors/wjx_20240523_04", delay=1),
            DataItem("alpha.wjx_20240425_01", path="factors/wjx_20240425_01", delay=1),
            DataItem("alpha.wjx_20240425_02", path="factors/wjx_20240425_02", delay=1),
            DataItem("alpha.wjx_20240425_03", path="factors/wjx_20240425_03", delay=1),
            DataItem("alpha.wjx_20240829_01", path="factors/wjx_20240829_01", delay=1),
            DataItem("alpha.wjx_20240829_02", path="factors/wjx_20240829_02", delay=1),
            DataItem("alpha.wjx_20240704_01", path="factors/wjx_20240704_01", delay=1),
            DataItem("alpha.wjx_20240704_02", path="factors/wjx_20240704_02", delay=1),
            DataItem("alpha.wjx_20240912_01", path="factors/wjx_20240912_01", delay=1),
            DataItem("alpha.wjx_20240912_02", path="factors/wjx_20240912_02", delay=1),
            DataItem("alpha.wjx_20240912_03", path="factors/wjx_20240912_03", delay=1),
            DataItem("alpha.wjx_20240411_01", path="factors/wjx_20240411_01", delay=1),
            DataItem("alpha.wjx_20240725_01", path="factors/wjx_20240725_01", delay=1),
            DataItem("alpha.xk_0224_02", path="factors/xk_0224_02", delay=1),
            DataItem("alpha.xk_0224_04", path="factors/xk_0224_04", delay=1),
            DataItem("alpha.xk_0224_07", path="factors/xk_0224_07", delay=1),
            DataItem("alpha.xk_0425_02", path="factors/xk_0425_02", delay=1),
            DataItem("alpha.xk_0202_02", path="factors/xk_0202_02", delay=1),
            DataItem("alpha.xk_0202_06", path="factors/xk_0202_06", delay=1),
            DataItem("alpha.xk_0202_08", path="factors/xk_0202_08", delay=1),
            DataItem("alpha.xk_0530_01", path="factors/xk_0530_01", delay=1),
            DataItem("alpha.xk_0530_02", path="factors/xk_0530_02", delay=1),
            DataItem("alpha.xk_0530_03", path="factors/xk_0530_03", delay=1),
            DataItem("alpha.xk_0411_02", path="factors/xk_0411_02", delay=1),
            DataItem("alpha.yz_20250219_01", path="factors/yz_20250219_01", delay=1),
            DataItem("alpha.yz_20250219_02", path="factors/yz_20250219_02", delay=1),
            DataItem("alpha.yz_20250219_03", path="factors/yz_20250219_03", delay=1),
            DataItem("alpha.yz_20250219_04", path="factors/yz_20250219_04", delay=1),
            DataItem("alpha.yz_20241106_01", path="factors/yz_20241106_01", delay=1),
            DataItem("alpha.yz_20241106_02", path="factors/yz_20241106_02", delay=1),
            DataItem("alpha.yz_20241030_01", path="factors/yz_20241030_01", delay=1),
            DataItem("alpha.yz_20241030_02", path="factors/yz_20241030_02", delay=1),
            DataItem("alpha.yz_20241030_03", path="factors/yz_20241030_03", delay=1),
            DataItem("alpha.yz_20241016_01", path="factors/yz_20241016_01", delay=1),
            DataItem("alpha.yz_20241016_02", path="factors/yz_20241016_02", delay=1),
            DataItem("alpha.yz_20241016_03", path="factors/yz_20241016_03", delay=1),
            DataItem("alpha.yz_20241016_04", path="factors/yz_20241016_04", delay=1),
            DataItem("alpha.yz_20240731_01", path="factors/yz_20240731_01", delay=1),
            DataItem("alpha.yz_20240731_02", path="factors/yz_20240731_02", delay=1),
            DataItem("alpha.yz_20240731_03", path="factors/yz_20240731_03", delay=1),
            DataItem("alpha.yz_20240731_04", path="factors/yz_20240731_04", delay=1),
            DataItem("alpha.yz_20240731_05", path="factors/yz_20240731_05", delay=1),
            DataItem("alpha.yz_20240731_06", path="factors/yz_20240731_06", delay=1),
            DataItem("alpha.yz_20240731_07", path="factors/yz_20240731_07", delay=1),
            DataItem("alpha.yz_20240620_01", path="factors/yz_20240620_01", delay=1),
            DataItem("alpha.yz_20240620_02", path="factors/yz_20240620_02", delay=1),
            DataItem("alpha.yz_20240620_03", path="factors/yz_20240620_03", delay=1),
            DataItem("alpha.yz_20240620_04", path="factors/yz_20240620_04", delay=1),
            DataItem("alpha.yz_20240620_05", path="factors/yz_20240620_05", delay=1),
            DataItem("alpha.yz_20240718_01", path="factors/yz_20240718_01", delay=1),
            DataItem("alpha.yz_20240718_02", path="factors/yz_20240718_02", delay=1),
            DataItem("alpha.yz_20240718_03", path="factors/yz_20240718_03", delay=1),
            DataItem("alpha.yz_20240718_04", path="factors/yz_20240718_04", delay=1),
            DataItem("alpha.yz_20241113_01", path="factors/yz_20241113_01", delay=1),
            DataItem("alpha.yz_20241113_02", path="factors/yz_20241113_02", delay=1),
            DataItem("alpha.yz_20241113_03", path="factors/yz_20241113_03", delay=1),
            DataItem("alpha.yz_20241113_04", path="factors/yz_20241113_04", delay=1),
            DataItem("alpha.yz_20250114_01", path="factors/yz_20250114_01", delay=1),
            DataItem("alpha.yz_20250114_02", path="factors/yz_20250114_02", delay=1),
            DataItem("alpha.yz_20240627_02", path="factors/yz_20240627_02", delay=1),
            DataItem("alpha.yz_20240704_01", path="factors/yz_20240704_01", delay=1),
            DataItem("alpha.yz_20240704_02", path="factors/yz_20240704_02", delay=1),
            DataItem("alpha.yz_20240704_03", path="factors/yz_20240704_03", delay=1),
            DataItem("alpha.yz_20240704_04", path="factors/yz_20240704_04", delay=1),
            DataItem("label", module="builtin.snap_label", delay=1),
            DataItem("execution", path=str(root / "1d_IntraVwap" / "IntraVwap.Vwap30.{ti:06d}")),
        )

    def model_input_sources(self):
        return (
            "alpha.cyz_20231130_01",
            "alpha.cyz_20231130_02",
            "alpha.cyz_20231130_03",
            "alpha.cyz_20240718_01",
            "alpha.cyz_20240718_02",
            "alpha.cyz_20240212_02",
            "alpha.cyz_20240212_01",
            "alpha.cyz_20240208_01",
            "alpha.cyz_20231026_01",
            "alpha.cyz_20231026_02",
            "alpha.cyz_20240627_1",
            "alpha.cyz_20240627_2",
            "alpha.cyz_20240627_3",
            "alpha.cyz_20240627_4",
            "alpha.cyz_20231019_01",
            "alpha.cyz_20231019_02",
            "alpha.cyz_20231207_01",
            "alpha.cyz_20231207_02",
            "alpha.cyz_20231207_03",
            "alpha.cyz_20231207_04",
            "alpha.cyz_20231207_05",
            "alpha.cyz_20240425_2",
            "alpha.cyz_20240425_3",
            "alpha.cyz_20240425_5",
            "alpha.cyz_20240425_4",
            "alpha.cyz_20240425_6",
            "alpha.cyz_20231123_01",
            "alpha.cyz_20240219_01",
            "alpha.cyz_20240220_02",
            "alpha.cyz_20240222_03",
            "alpha.cyz_20240226_04",
            "alpha.wjx_20231130_01",
            "alpha.wjx_20231130_02",
            "alpha.wjx_20240523_01",
            "alpha.wjx_20240523_02",
            "alpha.wjx_20240523_03",
            "alpha.wjx_20240523_04",
            "alpha.wjx_20240425_01",
            "alpha.wjx_20240425_02",
            "alpha.wjx_20240425_03",
            "alpha.wjx_20240829_01",
            "alpha.wjx_20240829_02",
            "alpha.wjx_20240704_01",
            "alpha.wjx_20240704_02",
            "alpha.wjx_20240912_01",
            "alpha.wjx_20240912_02",
            "alpha.wjx_20240912_03",
            "alpha.wjx_20240411_01",
            "alpha.wjx_20240725_01",
            "alpha.xk_0224_02",
            "alpha.xk_0224_04",
            "alpha.xk_0224_07",
            "alpha.xk_0425_02",
            "alpha.xk_0202_02",
            "alpha.xk_0202_06",
            "alpha.xk_0202_08",
            "alpha.xk_0530_01",
            "alpha.xk_0530_02",
            "alpha.xk_0530_03",
            "alpha.xk_0411_02",
            "alpha.yz_20250219_01",
            "alpha.yz_20250219_02",
            "alpha.yz_20250219_03",
            "alpha.yz_20250219_04",
            "alpha.yz_20241106_01",
            "alpha.yz_20241106_02",
            "alpha.yz_20241030_01",
            "alpha.yz_20241030_02",
            "alpha.yz_20241030_03",
            "alpha.yz_20241016_01",
            "alpha.yz_20241016_02",
            "alpha.yz_20241016_03",
            "alpha.yz_20241016_04",
            "alpha.yz_20240731_01",
            "alpha.yz_20240731_02",
            "alpha.yz_20240731_03",
            "alpha.yz_20240731_04",
            "alpha.yz_20240731_05",
            "alpha.yz_20240731_06",
            "alpha.yz_20240731_07",
            "alpha.yz_20240620_01",
            "alpha.yz_20240620_02",
            "alpha.yz_20240620_03",
            "alpha.yz_20240620_04",
            "alpha.yz_20240620_05",
            "alpha.yz_20240718_01",
            "alpha.yz_20240718_02",
            "alpha.yz_20240718_03",
            "alpha.yz_20240718_04",
            "alpha.yz_20241113_01",
            "alpha.yz_20241113_02",
            "alpha.yz_20241113_03",
            "alpha.yz_20241113_04",
            "alpha.yz_20250114_01",
            "alpha.yz_20250114_02",
            "alpha.yz_20240627_02",
            "alpha.yz_20240704_01",
            "alpha.yz_20240704_02",
            "alpha.yz_20240704_03",
            "alpha.yz_20240704_04",
        )

    def model_target(self):
        return "label", "label"
