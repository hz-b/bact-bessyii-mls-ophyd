import numpy as np
from typing import Dict
from bluesky.protocols import Reading
from event_model import DataKey
from ..raw.orbit import Orbit as ROrbit


_NUMERIC_COLS = ("X", "Y", "A", "B", "C", "D")
_MAX_BPM_NAME_LEN = 64


def table_bytes_to_str_dtype(descr):
    """Convert byte-string dtype descriptors (S) to unicode (U)."""
    return [(name, ttype.replace("S", "U")) for name, ttype in descr]


def table_to_structured_array(table) -> np.ndarray:
    """
    Convert a bluesky/ophyd Table into a numpy structured array.
    Kept here for reference but no longer used in read() directly —
    we split into flat columns for xarray compatibility.
    """
    raw_dtype = table.numpy_dtype()
    str_dtype = table_bytes_to_str_dtype(raw_dtype.descr)
    n_rows = len(table.BPM)
    data = np.empty(n_rows, dtype=str_dtype)
    for col_name, col_type in data.dtype.descr:
        col_values = getattr(table, col_name)
        if "U" in col_type:
            col_values = np.asarray(col_values).astype("U")
        data[col_name] = col_values
    return data


class Orbit(ROrbit):
    """
    Efficient orbit reader — Option C fixed.

    The structured array is built internally (preserving the data model)
    but stored as separate flat 1-D numpy arrays per column so that
    xarray/databroker can load them without a dimension mismatch.

    Each column becomes a clean (time, n_bpms) DataArray in the resulting
    xarray Dataset.

    BPM names are static within a run — they are emitted only on the first
    read() call to avoid storing ~31 KB of redundant string data on every
    event (~250 MB saved per 8000-event run).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bpm_names_emitted = False

    # ------------------------------------------------------------------
    # describe
    # ------------------------------------------------------------------

    async def describe(self) -> dict[str, DataKey]:
        d = await super().describe()

        tmp = d.pop(f"{self.name}-data")     # remove the raw table key
        n_bpms = tmp["shape"][0]             # first dim = number of BPMs

        # One DataKey per numeric column.
        for col in _NUMERIC_COLS:
            d[f"{self.name}-{col.lower()}"] = DataKey(
                source=tmp["source"],
                shape=[n_bpms],
                dtype="array",
                dtype_numpy="<f8",
            )

        # BPM names: described in schema but only emitted once in read().
        d[f"{self.name}-bpm-names"] = DataKey(
            source=tmp["source"],
            shape=[n_bpms],
            dtype="array",
            dtype_numpy=f"<U{_MAX_BPM_NAME_LEN}",
        )

        return d

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    async def read(self) -> Dict[str, Reading]:
        data = await super().read()

        t_data = data.pop(f"{self.name}-data")
        table  = t_data["value"]
        ts     = t_data["timestamp"]

        # Build the structured array internally — preserves the data model
        # and ensures all columns are processed consistently.
        structured = table_to_structured_array(table)

        # Split into flat columns for xarray compatibility.
        for col in _NUMERIC_COLS:
            data[f"{self.name}-{col.lower()}"] = Reading(
                timestamp=ts,
                value=structured[col],       # 1-D float array, zero-copy slice
            )

        # BPM names are static — emit only on the first read of each run.
        if not self._bpm_names_emitted:
            data[f"{self.name}-bpm-names"] = Reading(
                timestamp=ts,
                value=structured["BPM"],
            )
            self._bpm_names_emitted = True

        return data


__all__ = ["Orbit"]