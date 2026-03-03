from dataclasses import asdict
from typing import Dict

import numpy as np
from bluesky.protocols import Reading
from event_model import DataKey

from bact_device_models.devices.orbit import (
    BPMButtons,
    BPMPosition,
    BPMReading,
    Orbit as OrbitModel,
)

from ..raw.orbit import Orbit as ROrbit


class PPOrbit(ROrbit):
    """Provide read in data as orbit model"""

    async def describe(self) -> dict[str, DataKey]:
        d = await super().describe()
        d2 = {f"{self.name}-pos": DataKey(source="", shape=[], dtype="array")}
        d.update(d2)
        return d

    async def read(self) -> Dict[str, Reading]:
        data = await super().read()
        # todo: has ophyd / bluesky a helper func for splitting the read data?
        t_data = data[f"{self.name}-data"]
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
                    table.PosX, table.PosY,
                    table.ButtonA, table.ButtonB, table.ButtonC, table.ButtonD
                )
            ]
        )
        pos = {
            f"{self.name}-pos": Reading(
                timestamp=t_data["timestamp"], value=asdict(value)
            )
        }
        data.update(pos)
        return data
