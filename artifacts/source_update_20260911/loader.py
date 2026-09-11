from __future__ import annotations

from pathlib import Path
from comb2 import ComboDataLoader, DataItem
from comb2_simbase.cache_layout import ashare_cache_path


class ResearchLoader(ComboDataLoader):
    """Same-time daily factors and the researcher-indexed snapshot label."""

    def data_requirements(self):
        root = ashare_cache_path(self.config.cache_path)
        return (
            DataItem("alpha.yz_20250219_02", path="factors/yz_20250219_02", delay=1),
            DataItem("alpha.wjx_20240829_02", path="factors/wjx_20240829_02", delay=1),
            DataItem("alpha.guanxl_05", path="factors/guanxl_05", delay=1),
            DataItem("alpha.alpha1_20251008_01", path="factors/alpha1_20251008_01", delay=1),
            DataItem("alpha.alpha2_20251008_02", path="factors/alpha2_20251008_02", delay=1),
            DataItem("alpha.alpha3_20251008_03", path="factors/alpha3_20251008_03", delay=1),
            DataItem("alpha.alpha4_20251008_04", path="factors/alpha4_20251008_04", delay=1),
            DataItem("alpha.alpha5_20251008_05", path="factors/alpha5_20251008_05", delay=1),
            DataItem("label", module="builtin.snap_label", delay=1),
            DataItem("execution", path=str(root / "1d_IntraVwap" / "IntraVwap.Vwap30.{ti:06d}")),
        )

    def model_input_sources(self):
        return (
            "alpha.yz_20250219_02",
            "alpha.wjx_20240829_02",
            "alpha.guanxl_05",
            "alpha.alpha1_20251008_01",
            "alpha.alpha2_20251008_02",
            "alpha.alpha3_20251008_03",
            "alpha.alpha4_20251008_04",
            "alpha.alpha5_20251008_05",
        )

    def model_target(self):
        return "label", "label"
