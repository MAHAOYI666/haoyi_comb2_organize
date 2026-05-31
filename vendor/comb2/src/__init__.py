from __future__ import annotations

__all__ = [
    "ComboBase",
    "ComboBuffer",
    "ComboDataLoader",
    "ComboTrainDataset",
    "DefaultSelectionModule",
    "FeatureSpec",
    "LoaderConfig",
    "OpSpec",
    "SelectionModule",
    "SelectionPlan",
]


def __getattr__(name: str):
    if name == "ComboBase":
        from .ComboBase import ComboBase

        return ComboBase
    if name in {"ComboBuffer", "ComboDataLoader", "ComboTrainDataset", "FeatureSpec", "LoaderConfig", "OpSpec"}:
        from .DataLoader import ComboBuffer, ComboDataLoader, ComboTrainDataset, FeatureSpec, LoaderConfig, OpSpec

        return {
            "ComboBuffer": ComboBuffer,
            "ComboDataLoader": ComboDataLoader,
            "ComboTrainDataset": ComboTrainDataset,
            "FeatureSpec": FeatureSpec,
            "LoaderConfig": LoaderConfig,
            "OpSpec": OpSpec,
        }[name]
    if name in {"DefaultSelectionModule", "SelectionModule", "SelectionPlan"}:
        from .selection import DefaultSelectionModule, SelectionModule, SelectionPlan

        return {
            "DefaultSelectionModule": DefaultSelectionModule,
            "SelectionModule": SelectionModule,
            "SelectionPlan": SelectionPlan,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
