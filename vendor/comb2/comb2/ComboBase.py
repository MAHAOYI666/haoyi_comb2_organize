# Mengkang Li 2026/04/22

from __future__ import annotations

import datetime
import gc
import importlib.util
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from io import BytesIO
from typing import Any

import torch
import numpy as np

from .DataLoader import ComboDataLoader, ComboTrainDataset

ORGANIZE_ROOT = Path(__file__).resolve().parents[3]
if str(ORGANIZE_ROOT) not in sys.path:
    sys.path.insert(0, str(ORGANIZE_ROOT))
from vendor.perf_monitor import print_progress


class ComboBase:
    def __init__(self, node: Any):
        self.node = node
        self.model_path = node.model_path
        self.research_loader_path = getattr(node, "research_loader_path", None)
        self.research_dataset_path = getattr(node, "research_dataset_path", None)
        self.snaptime = node.snaptime
        self.livetrading = node.livetrading
        self.sample_times = tuple(node.sample_times)
        self.trainDelay = int(node.trainDelay)
        self.retDays = int(node.retDays)
        self.tsDays = int(node.tsDays)
        self.load_chunk_days = node.load_chunk_days
        self.seed = getattr(node, "seed", None)
        self.deterministic = bool(getattr(node, "deterministic", False))
        self.model_smooth_rate = node.model_smooth_rate
        self.model_keep_num = node.model_keep_num
        self.max_train_days = int(node.max_train_days)
        self.checkpoint_root = node.checkpoint_root
        self._set_random_seed()
        self.modelDir = os.path.join(self.checkpoint_root, self.snaptime) if self.checkpoint_root else None
        if self.modelDir:
            os.makedirs(self.modelDir, exist_ok=True)

        self.research_loader_cls = self._load_optional_research_class(
            self.research_loader_path,
            class_name="ResearchLoader",
            base_cls=ComboDataLoader,
            default_cls=ComboDataLoader,
        )
        self.research_dataset_cls = self._load_optional_research_class(
            self.research_dataset_path,
            class_name="ResearchDataset",
            base_cls=ComboTrainDataset,
            default_cls=ComboTrainDataset,
        )

        self.loader = self.research_loader_cls(node.loader_config)
        self.loader.monitor = getattr(node, "monitor", None)
        self.loader.set_point(int(node.start_ds), self.sample_times[0])
        self.model = None
        self.oldModel = None
        self.model_dt = -1
        self._last_train_check_ds: int | None = None
        self.alpha_history = node.alpha_history
        self.research_model_cls = self._load_research_model_class(self.model_path)

    def _set_random_seed(self):
        if self.seed == "":
            self.seed = None
        if self.seed is None:
            return
        self.seed = int(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

    def _release_torch_cache(self, tag: str):
        gc.collect()
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        print(f"[CUDA] tag={tag} cache_released")

    def _model_config(self) -> dict[str, Any]:
        model_config = dict(getattr(self.node, "model_config", {}))
        model_config.update(
            {
                "dtype": self.loader.dtype,
                "tsDays": self.tsDays,
                "num_features": self.loader.num_features,
                "feature_names": self.loader.feature_names,
                "sample_times": self.sample_times,
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
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if not hasattr(module, "ResearchModel"):
            raise AttributeError(f"{model_path} must define ResearchModel")
        return module.ResearchModel

    def _load_optional_research_class(self, path: str | None, *, class_name: str, base_cls: type, default_cls: type):
        if not path:
            return default_cls
        if not os.path.exists(path):
            raise FileNotFoundError(f"{class_name} file not found: {path}")
        module_name = f"comb2_{class_name}_{Path(path).stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if not hasattr(module, class_name):
            raise AttributeError(f"{path} must define {class_name}")
        cls = getattr(module, class_name)
        if not isinstance(cls, type) or not issubclass(cls, base_cls):
            raise TypeError(f"{class_name} in {path} must inherit from {base_cls.__name__}")
        return cls

    def Combine(self, di, ti):
        ds = self._resolve_date(di)
        ti = int(ti)
        assert ti in self.sample_times
        self.loader.set_point(ds, ti, refresh=self.livetrading)
        if self.livetrading:
            return self.CombineLive(ds, ti)
        return self.CombineHist(ds, ti)

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

    def _record_alpha(self, ds: int, ti: int | None = None):
        key = (int(ds), int(ti))
        if isinstance(self.node.alpha, torch.Tensor):
            self.alpha_history[key] = self.node.alpha.detach().cpu().clone()
        else:
            self.alpha_history[key] = self.node.alpha.copy()

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

    def _log_alpha(self, ds: int, ti: int | None, tag: str):
        ti_text = "" if ti is None else f" ti={ti}"
        print(f"[ALPHA] ds={ds}{ti_text} stage={tag} {self._summarize_alpha()}")

    def _buffer_ready(self, ds):
        return self.loader.date2didx(ds) - self.tsDays + 1 >= self.loader.data_start_didx

    def CombineLive(self, ds, ti):
        if self.modelDir:
            self.LoadCheckpointModel(self.modelDir, self._train_target_ds(ds))
        return self.GenComboPos(ds, ti)

    def CombineHist(self, ds, ti):
        if self.model is None and self.modelDir:
            self.LoadCheckpointModel(self.modelDir, self._train_target_ds(ds))
        alpha = self.GenComboPos(ds, ti)
        if self.needTrain(ds):
            self.Train(ds)
            if self.modelDir:
                self.SaveCheckpointModel(self.modelDir, self._train_target_ds(ds))
        return alpha

    def _resolve_date(self, di) -> int:
        if isinstance(di, int) and di in self.loader.mask.date:
            return di
        if isinstance(di, int) and 0 <= di < len(self.loader.mask.date):
            return self.loader.didx2date(di)
        return self.loader.align_date(int(di))

    def _prev_date(self, ds: int, offset: int = 1) -> int:
        didx = self.loader.date2didx(ds)
        return self.loader.didx2date(max(0, didx - offset))

    def _train_target_ds(self, ds):
        return self.loader.previous_date(ds, self.trainDelay)

    def _model_trainii(self, model) -> torch.Tensor | None:
        trainii = getattr(model, "trainii", None)
        if trainii is None:
            return None
        return torch.as_tensor(trainii, dtype=torch.long)

    def _predict_with_refill(self, model, feature_window, *, di, ti):
        trainii = self._model_trainii(model)
        window = feature_window if trainii is None else feature_window.index_select(1, trainii)
        pred = torch.as_tensor(model.predict(window, di=di, ti=ti), dtype=self.loader.dtype)
        if trainii is None:
            assert pred.shape == (len(self.loader.mask.code),)
            return pred
        assert pred.shape == (len(trainii),)
        full = torch.full((len(self.loader.mask.code),), torch.nan, dtype=self.loader.dtype)
        full[trainii] = pred.cpu()
        return full

    def GenComboPos(self, ds, ti):
        if self.model is None or not self._buffer_ready(ds):
            self._clear_alpha()
            self._record_alpha(ds, ti)
            return None
        window = self.loader.load_feature_window(ds, self.tsDays, ti)
        pred = self._predict_with_refill(self.model, window, di=ds, ti=ti)
        if self.oldModel is not None and self.model_smooth_rate < 1:
            old = self._predict_with_refill(self.oldModel, window, di=ds, ti=ti)
            pred = pred * self.model_smooth_rate + old * (1 - self.model_smooth_rate)
        self.node.alpha[:] = pred
        self._set_invalid_alpha(self.loader.gen_valid_mask(ds, ti))
        self._record_alpha(ds, ti)
        self._log_alpha(ds, ti, "predict")
        return self.node.alpha

    def Train(self, ds: int):
        target_ds = self._train_target_ds(ds)
        target_didx = self.loader.date2didx(target_ds)
        loading_days = target_didx - self.loader.data_start_didx + 1
        raw_ndays = loading_days
        ndays = min(loading_days, self.max_train_days)
        if ndays < self.tsDays + self.retDays - 1:
            raise ValueError(f"not enough training window for ds={ds}")

        if self.model is not None:
            model_data_in_memory = BytesIO()
            self.model.save(model_data_in_memory)
            model_data_in_memory.seek(0)
            self.oldModel = None
            self._release_torch_cache("before_old_model_replace")
            self.oldModel = self.research_model_cls(self._model_config())
            self.oldModel.load(model_data_in_memory)
            self._release_torch_cache("after_old_model_replace")

        print(
            f"[TRAIN] ds={ds} target_ds={target_ds} "
            f"loading_days={loading_days} raw_ndays={raw_ndays} "
            f"ndays={ndays} tsDays={self.tsDays}"
        )
        train_start = time.perf_counter()
        dataset_start = time.perf_counter()
        dataset = self.research_dataset_cls(
            self.loader,
            end_ds=target_ds,
            ndays=ndays,
            x_delay=self.retDays,
            ts_days=self.tsDays,
            load_chunk_days=self.load_chunk_days,
            codec=self.loader.codec,
        )
        dataset_time = time.perf_counter() - dataset_start
        print(f"[TRAIN] dataset_len={len(dataset)} valid_instruments={dataset.numValidinsts}")
        self.model = None
        self._release_torch_cache("before_new_model_fit")
        self.model = self.research_model_cls(self._model_config())
        fit_start = time.perf_counter()
        self.model.fit(dataset)
        fit_time = time.perf_counter() - fit_start
        dataset = None
        self._release_torch_cache("after_fit_dataset_release")
        if getattr(self.loader, "verbose", False):
            print_progress(
                f"Stage:Train ds={ds}",
                1,
                1,
                train_start,
                f"dataset {dataset_time:.2f}, fit {fit_time:.2f}",
                final=True,
            )
        print(f"[TRAIN] finished ds={ds} target_ds={target_ds}")

    def needTrain(self, ds: int) -> bool:
        if self._last_train_check_ds == int(ds):
            return False
        self._last_train_check_ds = int(ds)
        target_ds = self._train_target_ds(ds)
        if not self.isTrainDay(target_ds):
            return False
        if not self.modelDir:
            return True
        model_day = self.LoadCheckpointModel(self.modelDir, target_ds)
        return model_day != target_ds

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
        self.model = None
        self.oldModel = None
        self._release_torch_cache("before_checkpoint_load")
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
