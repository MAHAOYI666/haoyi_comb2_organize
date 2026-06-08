from __future__ import annotations

from pathlib import Path

import pandas as pd

from comb_eval.formatting import output_frame_to_text


def write_frame(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    if path.suffix == ".parquet":
        frame.to_parquet(path)
        return
    sep = "\t" if path.suffix in {".tsv", ".tab"} else ","
    frame.to_csv(path, sep=sep)


def write_text(text: str, path: str | Path) -> None:
    Path(path).write_text(text)


def frame_to_text(frame: pd.DataFrame) -> str:
    return output_frame_to_text(frame)


def select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[column for column in columns if column in frame.columns]]
