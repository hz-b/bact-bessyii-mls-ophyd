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

    BPM names are static within a run. They are still emitted on every
    event (required by event_model validation) but are also stored in
    read_configuration() so that downstream code can access them cheaply
    without scanning the full event stream.
    """

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

        # BPM names — must appear in every event (event_model requirement).
        d[f"{self.name}-bpm-names"] = DataKey(
            source=tmp["source"],
            shape=[n_bpms],
            dtype="array",
            dtype_numpy=f"<U{_MAX_BPM_NAME_LEN}",
        )

        return d

    # ------------------------------------------------------------------
    # describe_configuration / read_configuration
    # ------------------------------------------------------------------

    async def describe_configuration(self) -> dict[str, DataKey]:
        d = await super().describe_configuration()
        # Declare BPM names as configuration — stored once in the
        # descriptor document, not repeated in every event.
        t_data = (await super().read())[f"{self.name}-data"]
        table  = t_data["value"]
        n_bpms = len(table.BPM)
        d[f"{self.name}-bpm-names-config"] = DataKey(
            source=f"pva://ORBITCC:rdBpm",
            shape=[n_bpms],
            dtype="array",
            dtype_numpy=f"<U{_MAX_BPM_NAME_LEN}",
        )
        return d

    async def read_configuration(self) -> Dict[str, Reading]:
        d = await super().read_configuration()
        # Read BPM names once and store them as configuration.
        t_data = (await super().read())[f"{self.name}-data"]
        table  = t_data["value"]
        d[f"{self.name}-bpm-names-config"] = Reading(
            timestamp=t_data["timestamp"],
            value=np.asarray(table.BPM).astype("U"),
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

        # BPM names must be present on every event (event_model validation).
        # They are also stored cheaply in read_configuration() above.
        data[f"{self.name}-bpm-names"] = Reading(
            timestamp=ts,
            value=structured["BPM"],
        )

        return data


__all__ = ["Orbit"]