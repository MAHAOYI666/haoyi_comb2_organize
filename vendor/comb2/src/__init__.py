from __future__ import annotations

__all__ = [
    "ComboBase",
    "ComboBuffer",
    "ComboDataLoader",
    "ComboTrainDataset",
    "DataItem",
    "DataRegistry",
    "DefaultSelectionModule",
    "FeatureGroups",
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
    if name in {"ComboBuffer", "ComboDataLoader", "ComboTrainDataset", "FeatureGroups", "LoaderConfig"}:
        from .DataLoader import ComboBuffer, ComboDataLoader, ComboTrainDataset, FeatureGroups, LoaderConfig

        return {
            "ComboBuffer": ComboBuffer,
            "ComboDataLoader": ComboDataLoader,
            "ComboTrainDataset": ComboTrainDataset,
            "FeatureGroups": FeatureGroups,
            "LoaderConfig": LoaderConfig,
        }[name]
    if name in {"DataItem", "DataRegistry", "OpSpec", "Universe"}:
        from .DataRegistry import DataItem, DataRegistry, OpSpec, Universe

        return {
            "DataItem": DataItem,
            "DataRegistry": DataRegistry,
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
