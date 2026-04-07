from dataclasses import asdict
from typing import Dict

from bluesky.protocols import Reading
from event_model import DataKey

from bact_device_models.devices.orbit import (
    BPMButtons,
    BPMPosition,
    BPMReading,
    Orbit as OrbitModel,
)

from ..raw.orbit import Orbit as ROrbit
import json
import numpy as np


class Orbit(ROrbit):
    """Provide read in data as orbit model"""

    async def describe(self) -> dict[str, DataKey]:
        d = await super().describe()
        # Table not json serializable
        tmp = d.pop(f"{self.name}-data")
        descr_w_str = table_bytes_to_str_dtype(tmp["dtype_numpy"])
        descr_wo_str = table_dtype_without_str(tmp["dtype_numpy"])
        d2 = {
            # f"{self.name}-pos": DataKey(source=tmp["source"], shape=[], dtype="array"),
            # Todo: need to describe the data properly so that it fits into the
            # data model
             f"{self.name}-data": DataKey(
                  dtype="array",
                  source=tmp["source"],
                  shape=(len(descr_w_str),),# [len(descr_w_str)] + tmp["shape"],
                  dtype_numpy=[(key, type) for key, type in descr_w_str],
            ),
            f"{self.name}-data-sel": DataKey(
                  source=tmp["source"],
                  shape=[len(descr_wo_str)] + tmp["shape"],
                  dtype="array",
                 dtype_numpy=descr_wo_str,
             ),

        }
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


def table_without_str(table):
    """
    Todo:
        should only uniform type be supported
    """
    dtype = table.numpy_dtype()
    ndtype = table_dtype_without_str(dtype.descr)

    data = np.empty([len(table)], dtype=ndtype)

    for name, ttype in data.dtype.descr:
        tmp = getattr(table, name)
        if "U" in ttype:
            tmp = np.asarray(tmp).astype("U")
        data[name] = tmp
    return data


def table_dtype_without_str(descr):
    def valid_type(ttype: str) -> bool:
        if "S" in ttype:
            return False
        elif "U" in ttype:
            return False
        elif "i" in ttype:
            return False
        else:
            return True

    r = [(name, type) for name, type in descr if valid_type(type)]

    return r


__all__ = ["Orbit"]
