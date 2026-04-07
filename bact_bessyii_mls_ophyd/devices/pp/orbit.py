from typing import Dict

from bluesky.protocols import Reading
from event_model import DataKey


from ..raw.orbit import Orbit as ROrbit
import numpy as np


class Orbit(ROrbit):
    """Provide read in data as orbit model

    Todo:
        improve storage of normative table
    """

    async def describe(self) -> dict[str, DataKey]:
        d = await super().describe()
        # split up info in table in individual columns
        tmp = d.pop(f"{self.name}-data")
        descr_w_str = table_bytes_to_str_dtype(tmp["dtype_numpy"])
        d3 = {
            f"{self.name}-{name}": DataKey(
                dtype="array",
                source=tmp["source"],
                shape=tmp["shape"],
                dtype_numpy=type,
            )
            for name, type  in descr_w_str
        }
        d.update(d3)
        return d

    async def read(self) -> Dict[str, Reading]:
        data = await super().read()
        # todo: has ophyd / bluesky a helper func for splitting the read data?
        t_data = data.pop(f"{self.name}-data")
        table = t_data["value"]

        # Todo: this storage could be more efficient
        #       store it in this manner if it works
        #       currently everything is stored as a string
        converted_table = table_bytes_to_str(table)

        additional = {
            f"{self.name}-{name}": Reading(
                timestamp=t_data["timestamp"],
                value=converted_table[name]
            )
            for name, type  in converted_table.dtype.descr
        }

        data.update(additional)
        return data


def table_bytes_to_str_dtype(descr):
    return [(name, type.replace("S", "U")) for name, type in descr]


def table_bytes_to_str(table):
    """
    Todo:
        remove dependence on table.X
        find clean way to find its length
    """
    dtype = table.numpy_dtype()
    ndtype = table_bytes_to_str_dtype(dtype.descr)
    data = np.empty([len(table)], dtype=ndtype)
    for name, ttype in data.dtype.descr:
        tmp = getattr(table, name)
        if "U" in ttype:
            tmp = np.asarray(tmp).astype("U")
        data[name] = tmp
    return data


__all__ = ["Orbit"]
