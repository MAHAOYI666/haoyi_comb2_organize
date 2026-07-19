from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .np_utils import search_sorted_idx, search_sorted_side_idx


def _wants_dataframe(df_type: Any) -> bool:
    if df_type is True:
        return True
    if isinstance(df_type, str):
        return df_type.strip().lower() in {"true", "df", "dataframe", "frame", "pd"}
    return False


def _load_mmap(path, dtype, shape=None, offset=0):
    return np.memmap(filename=path, dtype=dtype, shape=shape, offset=offset, order="C", mode="r")


class MemmapArray:
    def __init__(self, memmaps):
        if not memmaps:
            raise ValueError("MemmapArray requires at least one block")
        self.dtype = memmaps[0].dtype
        self.ndim1 = memmaps[0].shape[1]
        self._memmaps = memmaps
        self.block_sizes = np.array([item.shape[0] for item in self._memmaps]).cumsum()

    @property
    def shape(self):
        return (int(self.block_sizes[-1]), self.ndim1)

    @property
    def values(self):
        return np.concatenate(self._memmaps, axis=0, casting="no")

    def _get_block_idx(self, index):
        if index < 0:
            index += int(self.block_sizes[-1])
        if index < 0 or index >= int(self.block_sizes[-1]):
            raise IndexError("MemmapArray index out of range")
        block_index = search_sorted_side_idx(self.block_sizes, index + 1, side="left")
        inner_index = index
        if block_index > 0:
            inner_index -= int(self.block_sizes[block_index - 1])
        return block_index, inner_index

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            block_index, inner_index = self._get_block_idx(int(index))
            return self._memmaps[block_index][[inner_index]]
        if isinstance(index, slice):
            start, stop, step = index.indices(int(self.block_sizes[-1]))
            if stop <= start:
                return np.empty((0, self.ndim1), dtype=self.dtype)
            res = []
            start_block, start_idx = self._get_block_idx(start)
            end_block, end_idx = self._get_block_idx(stop - 1)
            start_index, end_index = 0, None
            count = 0
            for block_idx in range(start_block, end_block + 1):
                if block_idx == start_block:
                    start_index = start_idx
                if block_idx == end_block:
                    end_index = end_idx + 1
                res.append(self._memmaps[block_idx][start_index:end_index:step])
                remain = (int(self.block_sizes[block_idx]) - count - start_index) % step
                start_index = step - remain if remain > 0 else 0
                count = int(self.block_sizes[block_idx])
            return np.concatenate(res, axis=0, casting="no")
        if isinstance(index, list):
            return np.concatenate([self[item] for item in index], axis=0, casting="no")
        raise TypeError("Index must be int, slice or list")

    def __array__(self, dtype=None):
        values = self.values
        return values.astype(dtype, copy=False) if dtype is not None else values

    def __len__(self):
        return int(self.block_sizes[-1])

    def __repr__(self):
        return f"MemmapArray(shape={self.shape}, dtype={self.dtype})"


class _SingleLocIndexer:
    def __init__(self, index):
        self._index = index

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            idx = search_sorted_idx(self._index, int(index))
            if idx is None:
                raise ValueError(f"{index} do not exist!")
            return idx
        if isinstance(index, slice):
            s_idx = search_sorted_side_idx(self._index, index.start, side="left") if index.start is not None else None
            e_idx = search_sorted_side_idx(self._index, index.stop, side="right") if index.stop is not None else None
            return slice(s_idx, e_idx, index.step)
        if isinstance(index, list):
            idx = [search_sorted_idx(self._index, item) for item in index]
            idx = [item for item in idx if item is not None]
            if not idx:
                raise ValueError(f"{index} do not exist")
            return idx
        raise TypeError("Index must be int, slice or list")


class _MultiLocIndexer:
    def __init__(self, index1, index2, idx_arr):
        self._index1 = index1
        self._index2 = index2
        self.idx_arr = idx_arr

    def _slice_idx(self, idx, side):
        if isinstance(idx, (int, np.integer)):
            pos = search_sorted_side_idx(self._index1, int(idx), side=side)
            if pos == self._index1.shape[0]:
                return int(np.nanmax(self.idx_arr[-1])) + 1
            return int(np.nanmin(self.idx_arr[pos]))
        if isinstance(idx, tuple):
            idx1 = search_sorted_idx(self._index1, idx[0])
            if idx1 is None:
                idx1 = search_sorted_side_idx(self._index1, idx[0], side=side)
                idx2 = 0
            else:
                idx2 = search_sorted_side_idx(self._index2, idx[1], side=side)
                if idx2 == len(self._index2) and side == "right":
                    idx1 += 1
                    idx2 = 0
                if idx1 == len(self._index1) and idx2 == 0:
                    return None
            return int(self.idx_arr[idx1, idx2])
        raise ValueError("Unsupported loc index type, only int or tuple")

    def _single_idx(self, idx):
        if isinstance(idx, (int, np.integer)):
            pos = search_sorted_idx(self._index1, int(idx))
            if pos is None:
                raise ValueError(f"{idx} do not exist!")
            return [int(item) for item in self.idx_arr[pos] if not np.isnan(item)]
        if isinstance(idx, tuple):
            idx1 = search_sorted_idx(self._index1, idx[0])
            idx2 = search_sorted_idx(self._index2, idx[1])
            if idx1 is None or idx2 is None:
                raise ValueError(f"{idx} do not exist!")
            return [int(self.idx_arr[idx1, idx2])]
        raise ValueError("Unsupported loc index type, only int or tuple")

    def __getitem__(self, index):
        if isinstance(index, slice):
            start = self._slice_idx(index.start, side="left") if index.start is not None else None
            stop = self._slice_idx(index.stop, side="right") if index.stop is not None else None
            return slice(start, stop, index.step)
        if isinstance(index, (int, np.integer, tuple)):
            return self._single_idx(index)
        if isinstance(index, list):
            idx = []
            for item in index:
                idx.extend(self._single_idx(item))
            if not idx:
                raise ValueError(f"{index} do not exist")
            return idx
        raise TypeError("Index must be int, tuple, slice or list")


class _LocDFIndexer:
    def __init__(self, idxer, frame):
        self._idxer = idxer
        self._frame = frame

    def __getitem__(self, index):
        idx = self._idxer[index]
        return pd.DataFrame(self._frame.values[idx], index=self._frame.index[idx], columns=self._frame.columns)


class _LocArrIndexer:
    def __init__(self, idxer, frame):
        self._idxer = idxer
        self._frame = frame

    def __getitem__(self, index):
        return self._frame.values[self._idxer[index]]


class MemmapDataFrame:
    def __init__(self, values: MemmapArray, index, columns, n_levels):
        self.values = values
        self.columns = columns
        self.index = index
        if n_levels == 1:
            self._idxer = _SingleLocIndexer(index)
        elif n_levels == 2:
            idx_frame = pd.Series(1, index=index).cumsum().unstack() - 1
            self._idxer = _MultiLocIndexer(idx_frame.index.values, idx_frame.columns.values, idx_frame.values)
        else:
            raise ValueError(f"Unsupported n_levels={n_levels}; only 1 and 2 are supported")
        self.dloc = _LocDFIndexer(self._idxer, self)
        self.aloc = _LocArrIndexer(self._idxer, self)

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype

    def __getitem__(self, index):
        return pd.DataFrame(self.values[index], index=self.index[index], columns=self.columns)

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return f"MemmapDataFrame(shape={self.shape}, dtype={self.dtype})"


class Memmaper2:
    def __init__(self, base_dir):
        self.base_dir = str(base_dir)
        if not Path(self.base_dir).exists():
            raise FileNotFoundError(f"[Errno 2] No such file or directory: {self.base_dir}")
        self._meta = np.load(f"{self.base_dir}/meta.npy", allow_pickle=True)
        self._index = np.load(f"{self.base_dir}/index.npy")
        self._columns = np.load(f"{self.base_dir}/columns.npy", allow_pickle=True)
        self._dtype = self._meta[0]()
        self._chunk_size = int(self._meta[-2])

    @property
    def chunks(self):
        return int(self._meta[-1])

    @property
    def shape(self):
        return tuple(int(item) for item in self._meta[2:-2])

    @property
    def dtype(self):
        return self._dtype

    @property
    def first_idx(self):
        n_levels = int(self._meta[1])
        if n_levels == 1:
            return int(self._index[0])
        if n_levels == 2:
            return int(self._index[1, 0]), int(self._index[0, np.nanargmin(self._index[0, 1:]) + 1])
        raise ValueError(f"unsupported index level: {n_levels}")

    @property
    def last_idx(self):
        n_levels = int(self._meta[1])
        if n_levels == 1:
            return int(self._index[-1])
        if n_levels == 2:
            return int(self._index[-1, 0]), int(self._index[0, np.nanargmax(self._index[-1, 1:]) + 1])
        raise ValueError(f"unsupported index level: {n_levels}")

    def __repr__(self):
        return f"DataFrame save by np.memmap, shape={self.shape}, chunk_size={self._chunk_size}, chunks={self.chunks}"

    def load_2d(self, start_ds=None, end_ds=None, df_type=False):
        start_ds = int(start_ds) if start_ds is not None else int(self._index[0])
        end_ds = int(end_ds) if end_ds is not None else int(self._index[-1])
        start_idx = search_sorted_side_idx(self._index, start_ds, side="left")
        end_idx = search_sorted_side_idx(self._index, end_ds, side="right")
        if end_idx <= start_idx:
            empty = np.empty((0, self.shape[-1]), dtype=self.dtype)
            data = MemmapArray([empty])
        else:
            data = self._load_2d_blocks(start_idx, end_idx)
        if _wants_dataframe(df_type):
            return MemmapDataFrame(data, index=self._index[start_idx:end_idx].astype(int), columns=self._columns, n_levels=1)
        return data

    def _load_2d_blocks(self, start_idx: int, end_idx: int) -> MemmapArray:
        start_chunk = start_idx // self._chunk_size
        start_chunk_idx = start_idx % self._chunk_size
        end_chunk = end_idx // self._chunk_size
        end_chunk_idx = end_idx % self._chunk_size
        data = []
        data_count = 0
        for offset_idx, chunk in enumerate(range(start_chunk, end_chunk + 1)):
            offset, shape = 0, None
            if offset_idx == 0:
                offset = start_chunk_idx * self.shape[-1] * self.dtype.itemsize
            if offset_idx == end_chunk - start_chunk:
                shape = (end_idx - start_idx - data_count, self.shape[-1])
                if end_chunk_idx == 0:
                    continue
            block = _load_mmap(f"{self.base_dir}/{chunk}.ares", dtype=self.dtype, offset=offset, shape=shape).reshape((-1, self.shape[-1]))
            data.append(block)
            data_count += block.shape[0]
        return MemmapArray(data)

    def load_3d(self, start_ds=None, start_time=None, end_ds=None, end_time=None, df_type=False):
        date_idx = self._index[1:, 0].astype(int)
        time_idx = self._index[0, 1:].astype(int)
        idx_arr = self._index[1:, 1:]
        start_ds = int(date_idx[0] if start_ds is None else start_ds)
        end_ds = int(date_idx[-1] if end_ds is None else end_ds)
        start_time = int(time_idx[0] if start_time is None else start_time)
        end_time = int(time_idx[-1] if end_time is None else end_time)

        start_ds_idx = search_sorted_side_idx(date_idx, start_ds, side="left")
        end_ds_idx = search_sorted_side_idx(date_idx, end_ds, side="right") - 1
        if end_ds_idx < start_ds_idx:
            empty = MemmapArray([np.empty((0, self.shape[-1]), dtype=self.dtype)])
            index = pd.MultiIndex.from_arrays([[], []])
            return MemmapDataFrame(empty, index=index, columns=self._columns, n_levels=2) if _wants_dataframe(df_type) else empty

        start_time_idx = search_sorted_side_idx(time_idx, start_time, side="left")
        end_time_idx = search_sorted_side_idx(time_idx, end_time, side="right")
        if date_idx[start_ds_idx] > start_ds:
            start_time_idx = 0
        if date_idx[end_ds_idx] < end_ds:
            end_time_idx = len(time_idx)

        start_chunk = start_ds_idx // self._chunk_size
        start_chunk_idx = start_ds_idx % self._chunk_size
        end_chunk = end_ds_idx // self._chunk_size
        data = []
        data_count = 0
        for offset_idx, chunk in enumerate(range(start_chunk, end_chunk + 1)):
            offset, shape = 0, None
            if offset_idx == 0:
                offset = np.nanmin(idx_arr[start_ds_idx, start_time_idx:]) - np.nanmin(idx_arr[start_ds_idx - start_chunk_idx])
                offset = int(offset) * self.shape[-1] * self.dtype.itemsize
            if offset_idx == end_chunk - start_chunk:
                shape_rows = np.nanmax(idx_arr[end_ds_idx, :end_time_idx]) - np.nanmin(idx_arr[start_ds_idx, start_time_idx:]) - data_count + 1
                if shape_rows == 0:
                    continue
                shape = (int(shape_rows), self.shape[-1])
            block = _load_mmap(f"{self.base_dir}/{chunk}.ares", dtype=self.dtype, offset=int(offset), shape=shape).reshape((-1, self.shape[-1]))
            data.append(block)
            data_count += block.shape[0]
        values = MemmapArray(data)
        if _wants_dataframe(df_type):
            index_frame = pd.DataFrame(idx_arr[start_ds_idx : end_ds_idx + 1].copy(), index=date_idx[start_ds_idx : end_ds_idx + 1], columns=time_idx)
            index_frame.iloc[0, :start_time_idx] = float("nan")
            index_frame.iloc[-1, end_time_idx:] = float("nan")
            return MemmapDataFrame(values, index=index_frame.stack().index, columns=self._columns, n_levels=2)
        return values

    def load(self, start_ds=None, end_ds=None, df_type=False, **kwargs):
        n_levels = int(self._meta[1])
        if n_levels == 1:
            return self.load_2d(start_ds=start_ds, end_ds=end_ds, df_type=df_type)
        if n_levels == 2:
            return self.load_3d(start_ds=start_ds, end_ds=end_ds, df_type=df_type, **kwargs)
        raise ValueError(f"DataFrame with index level = {n_levels}, do not support in current load method")
