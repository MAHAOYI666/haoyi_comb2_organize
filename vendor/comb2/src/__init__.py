from __future__ import annotations

__all__ = [
    "ComboBase",
    "ComboBuffer",
    "ComboDataLoader",
    "ComboTrainDataset",
    "DataItem",
    "DataRegistry",
    "DefaultSelectionModule",
    "FeatureSpec",
    "LoaderConfig",
    "OpSpec",
    "SelectionModule",
    "SelectionPlan",
    "Universe",
]


def __getattr__(name: str):
    if name == "ComboBase":
        from .ComboBase import ComboBase

        return ComboBase
    if name in {"ComboBuffer", "ComboDataLoader", "ComboTrainDataset", "LoaderConfig"}:
        from .DataLoader import ComboBuffer, ComboDataLoader, ComboTrainDataset, LoaderConfig

        return {
            "ComboBuffer": ComboBuffer,
            "ComboDataLoader": ComboDataLoader,
            "ComboTrainDataset": ComboTrainDataset,
            "LoaderConfig": LoaderConfig,
        }[name]
    if name in {"DataItem", "DataRegistry", "FeatureSpec", "OpSpec", "Universe"}:
        from .DataRegistry import DataItem, DataRegistry, FeatureSpec, OpSpec, Universe

        return {
            "DataItem": DataItem,
            "DataRegistry": DataRegistry,
            "FeatureSpec": FeatureSpec,
            "OpSpec": OpSpec,
            "Universe": Universe,
        }[name]
    if name in {"DefaultSelectionModule", "SelectionModule", "SelectionPlan"}:
        from .selection import DefaultSelectionModule, SelectionModule, SelectionPlan

        return {
            "DefaultSelectionModule": DefaultSelectionModule,
            "SelectionModule": SelectionModule,
            "SelectionPlan": SelectionPlan,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
