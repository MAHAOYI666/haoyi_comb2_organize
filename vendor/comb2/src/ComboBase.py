# Mengkang Li 2026/04/22

from __future__ import annotations

import datetime
import importlib.util
import os
import re
import shutil
from io import BytesIO
from typing import Any

import torch

from .DataLoader import ComboBuffer, ComboDataLoader, ComboTrainDataset, LoaderConfig, nan_to_num
from .selection import DefaultSelectionModule


class ComboBase:
    def __init__(self, node: Any):
        self.node = node
        self.model_path = node.model_path
        self.snaptime = node.snaptime
        self.livetrading = node.livetrading
        self.trainDelay = node.trainDelay
        self.retDays = node.retDays
        self.tsDays = node.tsDays
        self.model_smooth_rate = node.model_smooth_rate
        self.model_keep_num = node.model_keep_num
        self.select_days = node.select_days
        self.max_train_days = int(node.max_train_days)
        self.checkpoint_root = node.checkpoint_root
        self.modelDir = os.path.join(self.checkpoint_root, self.snaptime) if self.checkpoint_root else None
        if self.modelDir:
            os.makedirs(self.modelDir, exist_ok=True)

        self.loader = ComboDataLoader(node.loader_config)
        self.loader.monitor = getattr(node, "monitor", None)
        self.buffer = ComboBuffer(feat_size=self.loader.num_features, keepdays=self.tsDays, instsz=len(self.loader.mask.code), dtype=self.loader.dtype)
        self.selection = node.selection_module or DefaultSelectionModule(
            max_train_days=self.max_train_days,
            select_days=self.select_days,
        )
        self.model = None
        self.oldModel = None
        self.model_dt = -1
        self.reset_buffer = True
        self.alpha_history = node.alpha_history
        self.research_model_cls = self._load_research_model_class(self.model_path)

    def _model_config(self) -> dict[str, Any]:
        model_config = dict(getattr(self.node, "model_config", {}))
        model_config.update(
            {
                "dtype": self.loader.dtype,
                "tsDays": self.tsDays,
                "num_features": self.loader.num_features,
            }
        )
        if "hiddenSize" in model_config:
            model_config.setdefault("hidden_size", model_config["hiddenSize"])
        if "fcSize" in model_config:
            model_config.setdefault("fc_size", model_config["fcSize"])
        if "batchSize" in model_config:
            model_config.setdefault("batch_size", model_config["batchSize"])
        return model_config

    def _load_research_model_class(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"research model file not found: {model_path}")
        spec = importlib.util.spec_from_file_location("comb2_research_model", model_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        if not hasattr(module, "ResearchModel"):
            raise AttributeError(f"{model_path} must define ResearchModel")
        return module.ResearchModel

    def Combine(self, di, ti=None):
        if self.livetrading:
            self.CombineLive(di, ti)
        else:
            self.CombineHist(di, ti)

    def _clear_alpha(self):
        if isinstance(self.node.alpha, torch.Tensor):
            self.node.alpha.zero_()
        else:
            self.node.alpha[:] = 0

    def _set_invalid_alpha(self, valid_mask: torch.Tensor):
        if isinstance(self.node.alpha, torch.Tensor) and self.node.alpha.is_floating_point():
            self.node.alpha[~valid_mask] = torch.nan
        else:
            self.node.alpha[~valid_mask.cpu().numpy()] = float("nan")

    def _record_alpha(self, ds: int):
        if isinstance(self.node.alpha, torch.Tensor):
            self.alpha_history[int(ds)] = self.node.alpha.detach().cpu().clone()
        else:
            self.alpha_history[int(ds)] = self.node.alpha.copy()

    def _summarize_alpha(self) -> str:
        alpha = self.node.alpha
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.as_tensor(alpha)
        alpha32 = alpha.detach().to(torch.float32)
        finite_mask = torch.isfinite(alpha32)
        valid = alpha32[finite_mask]
        if valid.numel() == 0:
            return "all values are NaN"
        nonzero = (valid != 0).sum().item()
        return (
            f"count={valid.numel()} "
            f"nonzero={nonzero} "
            f"mean={valid.mean().item():.6f} "
            f"std={valid.std(unbiased=False).item():.6f} "
            f"min={valid.min().item():.6f} "
            f"max={valid.max().item():.6f}"
        )

    def _log_alpha(self, ds: int, tag: str):
        print(f"[ALPHA] ds={ds} stage={tag} {self._summarize_alpha()}")

    def _buffer_ready(self, ds: int) -> bool:
        end_didx = self.loader.date2didx(ds)
        return end_didx - self.tsDays + 1 >= self.loader.data_start_didx

    def CombineLive(self, di, ti=None):
        ds = self._resolve_date(di)
        pred_ds = self._prev_date(ds)
        if self.modelDir:
            self.LoadCheckpointModel(self.modelDir, pred_ds)
        self.GenComboPos(pred_ds)

    def CombineHist(self, di, ti=None):
        ds = self._resolve_date(di)
        pred_ds = self._prev_date(ds)
        if self.model is None and self.modelDir:
            self.LoadCheckpointModel(self.modelDir, pred_ds)
        self.GenComboPos(pred_ds)
        if self.needTrain(ds):
            self.Train(ds)
            if self.modelDir:
                self.SaveCheckpointModel(self.modelDir, self._prev_date(ds, self.trainDelay))

    def _resolve_date(self, di) -> int:
        if isinstance(di, int) and di in self.loader.mask.date:
            return di
        if isinstance(di, int) and 0 <= di < len(self.loader.mask.date):
            return self.loader.didx2date(di)
        return self.loader.align_date(int(di))

    def _prev_date(self, ds: int, offset: int = 1) -> int:
        didx = self.loader.date2didx(ds)
        return self.loader.didx2date(max(0, didx - offset))

    def buffer_load(self, ds: int):
        end_didx = self.loader.date2didx(ds)
        if self.reset_buffer:
            loaddays = self.tsDays
            self.reset_buffer = False
        else:
            loaddays = 1
        for dd in range(loaddays):
            didx = end_didx - dd
            feature = self.loader.gen_feature(self.loader.didx2date(didx))
            self.buffer.append(feature, didx)

    def _model_trainii(self, model) -> torch.Tensor | None:
        trainii = getattr(model, "trainii", None)
        if trainii is None:
            return None
        return torch.as_tensor(trainii, dtype=torch.long)

    def _predict_with_refill(self, model, feature_window: torch.Tensor) -> torch.Tensor:
        trainii = self._model_trainii(model)
        if trainii is None:
            pred = self.predict(model, feature_window)
            if pred.numel() == len(self.loader.mask.code):
                return pred.to(dtype=self.loader.dtype)
            full_pred = torch.full((len(self.loader.mask.code),), torch.nan, dtype=self.loader.dtype)
            full_pred[:pred.numel()] = pred.to(dtype=self.loader.dtype)
            return full_pred
        pred = self.predict(model, feature_window[:, trainii])
        full_pred = torch.full((len(self.loader.mask.code),), torch.nan, dtype=self.loader.dtype)
        full_pred[trainii] = pred.to(dtype=self.loader.dtype)
        return full_pred

    def GenComboPos(self, ds: int):
        if self.model is None:
            self._clear_alpha()
            self._record_alpha(ds)
            self._log_alpha(ds, "no_model")
            return None
        if not self._buffer_ready(ds):
            self._clear_alpha()
            self._record_alpha(ds)
            self._log_alpha(ds, "buffer_warmup")
            return None
        self.buffer_load(ds)
        end_didx = self.loader.date2didx(ds)
        didx_list = [end_didx - (self.tsDays - 1) + i for i in range(self.tsDays)]
        feature_window = nan_to_num(self.buffer.get(didx_list), 0.0).to(self.loader.dtype)
        cur_pred = self._predict_with_refill(self.model, feature_window)
        if self.oldModel is not None:
            old_pred = self._predict_with_refill(self.oldModel, feature_window)
            cur_pred = cur_pred * self.model_smooth_rate + old_pred * (1 - self.model_smooth_rate)
        valid_mask = self.loader.gen_valid_mask(ds)
        self.node.alpha[:] = cur_pred
        self._set_invalid_alpha(valid_mask)
        self._record_alpha(ds)
        self._log_alpha(ds, "predict")
        return self.node.alpha

    def predict(self, model, feature_window: torch.Tensor) -> torch.Tensor:
        pred = model.predict(feature_window)
        if not isinstance(pred, torch.Tensor):
            pred = torch.as_tensor(pred, dtype=self.loader.dtype)
        return pred.to(dtype=self.loader.dtype)

    def Train(self, ds: int):
        plan = self.selection.build_train_plan(
            ds,
            loader=self.loader,
            context={
                "trainDelay": self.trainDelay,
                "retDays": self.retDays,
                "tsDays": self.tsDays,
                "prev_date": self._prev_date,
            },
        )

        if self.model is not None:
            model_data_in_memory = BytesIO()
            self.model.save(model_data_in_memory)
            model_data_in_memory.seek(0)
            self.oldModel = self.research_model_cls(self._model_config())
            self.oldModel.load(model_data_in_memory)

        self.reset_buffer = True
        print(
            f"[TRAIN] ds={ds} target_ds={plan.target_ds} "
            f"loading_days={plan.loading_days} raw_ndays={plan.raw_ndays} ndays={plan.ndays} tsDays={self.tsDays}"
        )
        dataset = ComboTrainDataset(
            self.loader,
            end_ds=plan.target_ds,
            ndays=plan.ndays,
            x_delay=self.retDays,
            step_size=self.tsDays,
            validinsts=plan.validinsts,
        )
        plan.validinsts = dataset.validinsts
        self.selection.before_fit(dataset, plan)
        print(f"[TRAIN] dataset_len={len(dataset)} valid_instruments={dataset.numValidinsts}")
        self.model = self.research_model_cls(self._model_config())
        self.model.fit(dataset)
        self.selection.after_fit(self.model, plan)
        print(f"[TRAIN] finished ds={ds} target_ds={plan.target_ds}")

    def needTrain(self, ds: int) -> bool:
        target_ds = self._prev_date(ds, self.trainDelay)
        train_day = self.isTrainDay(target_ds)
        if not self.modelDir:
            print(f"[TRAIN-CHECK] ds={ds} target_ds={target_ds} checkpoint=disabled train_day={train_day} -> {train_day}")
            return train_day
        model_day = self.LoadCheckpointModel(self.modelDir, target_ds)
        if model_day is False:
            print(f"[TRAIN-CHECK] ds={ds} target_ds={target_ds} model_day=None train_day={train_day} -> {train_day}")
            return train_day
        if model_day == target_ds:
            print(f"[TRAIN-CHECK] ds={ds} target_ds={target_ds} model_day={model_day} exact_match=True -> False")
            return False
        model_didx = self.loader.date2didx(model_day)
        target_didx = self.loader.date2didx(target_ds)
        outdated = target_didx - model_didx > 30
        decision = outdated
        print(
            f"[TRAIN-CHECK] ds={ds} target_ds={target_ds} model_day={model_day} "
            f"train_day={train_day} outdated={outdated} -> {decision}"
        )
        return decision

    def isTrainDay(self, ds: int) -> bool:
        didx = self.loader.date2didx(ds)
        if didx >= len(self.loader.mask.date) - 1:
            return True

        def is_trading_week_end(cur_didx: int) -> bool:
            if cur_didx >= len(self.loader.mask.date) - 1:
                return True
            today = datetime.datetime.strptime(str(self.loader.didx2date(cur_didx)), "%Y%m%d")
            next_day = datetime.datetime.strptime(str(self.loader.didx2date(cur_didx + 1)), "%Y%m%d")
            return (next_day - today).days > 1

        def next_trading_week_end(cur_didx: int) -> int:
            upper = min(cur_didx + 10, len(self.loader.mask.date) - 1)
            for next_idx in range(cur_didx + 1, upper):
                if is_trading_week_end(next_idx):
                    return self.loader.didx2date(next_idx)
            return self.loader.didx2date(upper)

        today = str(ds)
        month = today[4:6]
        next_week_end_date = str(next_trading_week_end(didx))
        if is_trading_week_end(didx):
            return next_week_end_date[4:6] != month
        return False

    def LoadCheckpointModel(self, save_dir: str, dt: int):
        if not save_dir or not os.path.exists(save_dir):
            return False
        matching_dirs = [x for x in os.listdir(save_dir) if re.fullmatch(r"\d{8}", x)]
        matching_dirs.sort()
        if not matching_dirs:
            return False
        model_dates = [int(x) for x in matching_dirs]
        valid_dates = [x for x in model_dates if x <= dt]
        if not valid_dates:
            return False
        model_date_to_use = valid_dates[-1]
        if model_date_to_use == self.model_dt:
            return model_date_to_use
        model_dir = os.path.join(save_dir, str(model_date_to_use))
        model_path = os.path.join(model_dir, "model")
        if not os.path.exists(model_path):
            return False
        self.model = self.research_model_cls(self._model_config())
        self.model.load(model_path)
        old_model_path = os.path.join(model_dir, "oldmodel")
        if os.path.exists(old_model_path):
            self.oldModel = self.research_model_cls(self._model_config())
            self.oldModel.load(old_model_path)
        else:
            self.oldModel = None
        self.model_dt = model_date_to_use
        self.diskclean(self.model_keep_num)
        return model_date_to_use

    def SaveCheckpointModel(self, save_dir: str, dt: int):
        if self.model is None or not save_dir or self.model_keep_num == 0:
            return
        dt_dir = os.path.join(save_dir, str(dt))
        os.makedirs(dt_dir, exist_ok=True)
        self.model.save(os.path.join(dt_dir, "model"))
        if self.oldModel is not None:
            self.oldModel.save(os.path.join(dt_dir, "oldmodel"))
        self.model_dt = dt
        self.diskclean(self.model_keep_num)

    def diskclean(self, keep_num: int = 4):
        if keep_num < 0:
            return
        if keep_num == 0:
            return
        if not self.modelDir or not os.path.exists(self.modelDir):
            return
        matching_dirs = [os.path.join(self.modelDir, x) for x in os.listdir(self.modelDir) if re.fullmatch(r"\d{8}", x)]
        matching_dirs.sort()
        while len(matching_dirs) > keep_num:
            shutil.rmtree(matching_dirs.pop(0), ignore_errors=True)
