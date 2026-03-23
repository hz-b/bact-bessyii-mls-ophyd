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
    We split into flat columns for xarray compatibility in read(),
    but building the structured array first ensures all columns are
    processed consistently in one place.
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
    Efficient orbit reader — Option C final.

    The structured array is built internally (preserving the data model)
    but stored as separate flat 1-D numpy arrays per column so that
    xarray/databroker can load them without a dimension mismatch.

    Each numeric column becomes a clean (time, n_bpms) DataArray in the
    resulting xarray Dataset.

    BPM names are static within a run and are intentionally kept OUT of
    describe() so that event_model never requires them in every event
    document. Instead they are stored exactly once in read_configuration(),
    which bluesky calls once at the start of each run and writes into the
    configuration document — not the event stream.

    Access after the run:
        run.primary.config['orb']['orb-bpm-names']
    """

    # ------------------------------------------------------------------
    # describe — numeric columns only, no BPM names
    # ------------------------------------------------------------------

    async def describe(self) -> dict[str, DataKey]:
        d = await super().describe()

        tmp = d.pop(f"{self.name}-data")     # remove the raw table key
        n_bpms = tmp["shape"][0]             # first dim = number of BPMs

        # Numeric columns only — BPM names deliberately excluded so they
        # are never demanded in every event by event_model validation.
        for col in _NUMERIC_COLS:
            d[f"{self.name}-{col.lower()}"] = DataKey(
                source=tmp["source"],
                shape=[n_bpms],
                dtype="array",
                dtype_numpy="<f8",
            )

        return d

    # ------------------------------------------------------------------
    # describe_configuration / read_configuration — BPM names stored once
    # ------------------------------------------------------------------

    async def describe_configuration(self) -> dict[str, DataKey]:
        d = await super().describe_configuration()
        t_data = (await super().read())[f"{self.name}-data"]
        n_bpms = len(t_data["value"].BPM)
        d[f"{self.name}-bpm-names"] = DataKey(
            source=f"pva://ORBITCC:rdBpm",
            shape=[n_bpms],
            dtype="array",
            dtype_numpy=f"<U{_MAX_BPM_NAME_LEN}",
        )
        return d

    async def read_configuration(self) -> Dict[str, Reading]:
        d = await super().read_configuration()
        t_data = (await super().read())[f"{self.name}-data"]
        table  = t_data["value"]
        # BPM names written once into the configuration document.
        d[f"{self.name}-bpm-names"] = Reading(
            timestamp=t_data["timestamp"],
            value=np.asarray(table.BPM).astype("U"),
        )
        return d

    # ------------------------------------------------------------------
    # read — numeric columns only, zero BPM name overhead per event
    # ------------------------------------------------------------------

    async def read(self) -> Dict[str, Reading]:
        data = await super().read()

        t_data = data.pop(f"{self.name}-data")
        table  = t_data["value"]
        ts     = t_data["timestamp"]

        structured = table_to_structured_array(table)

        for col in _NUMERIC_COLS:
            data[f"{self.name}-{col.lower()}"] = Reading(
                timestamp=ts,
                value=structured[col],       # zero-copy slice
            )

        return data


__all__ = ["Orbit"]