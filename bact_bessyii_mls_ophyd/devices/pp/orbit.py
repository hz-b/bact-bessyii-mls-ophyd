import numpy as np
from typing import Dict
from bluesky.protocols import Reading
from event_model import DataKey
from ..raw.orbit import Orbit as ROrbit


def table_bytes_to_str_dtype(descr):
    """Convert byte-string dtype descriptors (S) to unicode (U)."""
    return [(name, ttype.replace("S", "U")) for name, ttype in descr]


def table_to_structured_array(table) -> np.ndarray:
    """
    Convert a bluesky/ophyd Table into a numpy structured array.

    Byte-string columns (S) are promoted to unicode (U) so the result
    is JSON-serialisable and round-trips cleanly through tiled / databroker.
    All numeric columns are kept as-is — no copies beyond what numpy needs
    to build the structured array.
    """
    raw_dtype = table.numpy_dtype()          # numpy dtype of the Table
    str_dtype  = table_bytes_to_str_dtype(raw_dtype.descr)

    n_rows = len(table.BPM)                  # number of BPMs
    data   = np.empty(n_rows, dtype=str_dtype)

    for col_name, col_type in data.dtype.descr:
        col_values = getattr(table, col_name)
        if "U" in col_type:                  # byte → unicode, once per col
            col_values = np.asarray(col_values).astype("U")
        data[col_name] = col_values

    return data


class Orbit(ROrbit):
    """
    Efficient orbit reader.

    Stores each reading as a single numpy structured array that contains
    all columns (BPM, X, Y, A, B, C, D).  This avoids the expensive
    Python-level loop + dataclass + asdict() round-trip of the previous
    implementation and keeps all numeric data in contiguous memory.
    """

    # ------------------------------------------------------------------
    # describe
    # ------------------------------------------------------------------

    async def describe(self) -> dict[str, DataKey]:
        d = await super().describe()

        # Remove the raw table key — we replace it with a structured array.
        tmp = d.pop(f"{self.name}-data")

        # Build the dtype once here so describe() is self-consistent.
        # shape[0] = number of BPMs; the structured array is 1-D.
        d[f"{self.name}-data"] = DataKey(
            source=tmp["source"],
            shape=tmp["shape"][:1],          # [n_bpms]  (drop inner dims)
            dtype="array",
            dtype_numpy=table_bytes_to_str_dtype(tmp["dtype_numpy"]),
        )

        return d

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    async def read(self) -> Dict[str, Reading]:
        data = await super().read()

        t_data = data.pop(f"{self.name}-data")
        table  = t_data["value"]

        structured = table_to_structured_array(table)

        data[f"{self.name}-data"] = Reading(
            timestamp=t_data["timestamp"],
            value=structured,                # numpy structured array
        )

        return data


__all__ = ["Orbit"]