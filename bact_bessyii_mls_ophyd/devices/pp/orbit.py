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
        d2 = {
            f"{self.name}-pos": DataKey(source=tmp["source"], shape=[], dtype="array"),
            # Todo: need to describe the data properly so that it fits into the
            # data model
            # f"{self.name}-data": DataKey(
            #     source=tmp["source"],
            #     shape=[9] + tmp["shape"],
            #     dtype="array",
            #     dtype_numpy=table_bytes_to_str_dtype(tmp["dtype_numpy"]),
            # ),
        }
        d.update(d2)
        return d

    async def read(self) -> Dict[str, Reading]:
        data = await super().read()
        # todo: has ophyd / bluesky a helper func for splitting the read data?
        t_data = data.pop(f"{self.name}-data")
        table = t_data["value"]
        value = OrbitModel(
            orbit=[
                BPMReading(
                    name=name,
                    pos=BPMPosition(x, y),
                    btns=BPMButtons(a, b, c, d),
                )
                for name, x, y, a, b, c, d in zip(
                    table.BPM,
                    table.PosX,
                    table.PosY,
                    table.ButtonA,
                    table.ButtonB,
                    table.ButtonC,
                    table.ButtonD,
                )
            ]
        )

        # Todo: this storage could be more efficient
        #       store it in this manner if it works
        #       currently everything is stored as a string
        # additional = {
        #     f"{self.name}-pos": Reading(
        #         timestamp=t_data["timestamp"], value=asdict(value)
        #     ),
        #     # can it store it if it was a numpy table?
        #     f"{self.name}-data": Reading(
        #         timestamp=t_data["timestamp"], value=table_bytes_to_str(table)
        #     ),
        # }
        # data.update(additional)
        return data


def table_bytes_to_str_dtype(descr):
    return [(name, type.replace("S", "U")) for name, type in descr]


def table_bytes_to_str(table):
    dtype = table.numpy_dtype()
    ndtype = table_bytes_to_str_dtype(dtype.descr)
    data = np.empty([len(table.PosX)], dtype=ndtype)
    for name, ttype in data.dtype.descr:
        tmp = getattr(table, name)
        if "U" in ttype:
            tmp = np.asarray(tmp).astype("U")
        data[name] = tmp
    return data


__all__ = ["Orbit"]
