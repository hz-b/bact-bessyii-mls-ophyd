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
        d.pop(f"{self.name}-btns")
        rpos = d.pop(f"{self.name}-rpos")
        (L,) = rpos["shape"]
        assert L % 2 == 0
        d2 = {f"{self.name}-pos": DataKey(source="", shape=[], dtype="array")}
        d.update(d2)
        return d

    async def read(self) -> Dict[str, Reading]:
        data = await super().read()
        # todo: has ophyd / bluesky a helper func for splitting the read data?
        pos_pkg = data.pop(f"{self.name}-rpos")
        btn_pkg = data.pop(f"{self.name}-btns")
        pos = np.reshape(pos_pkg["value"], (-1, 2))
        btns = np.reshape(btn_pkg["value"], (-1, 4))
        names = await self.names.get_value()
        value = OrbitModel(
            orbit=[
                BPMReading(
                    name=name,
                    pos=BPMPosition(*p),
                    btns=BPMButtons(*b),
                )
                for name, p, b in zip(names, pos, btns)
            ]
        )
        pos = {
            f"{self.name}-pos": Reading(
                timestamp=pos_pkg["timestamp"], value=asdict(value)
            )
        }
        data.update(pos)
        return data
