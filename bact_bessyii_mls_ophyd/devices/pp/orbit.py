from typing import Dict
import jsons

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

import logging

logger = logging.getLogger("bact-bessyii-mls-ophyd")


class PPOrbit(ROrbit):
    """Provide read in data as orbit model"""

    async def describe(self) -> dict[str, DataKey]:
        d = await super().describe()
        d.pop(f"{self.name}-btns")
        # d.pop(f"{self.name}_names")
        rpos = d.pop(f"{self.name}-rpos")
        L, = rpos["shape"]
        assert L % 2 == 0
        d2 = {f"{self.name}-pos" : DataKey(source="", shape=[L//2], dtype="array")}
        d.update(d2)
        return d

    async def read(self) -> Dict[str, Reading]:
        data = await super().read()
        data_names = await self.names.read()
        # todo: has ophyd / bluesky a helper func for splitting the read data?

        # logger.warning(f"{list(data)}")
        pos_pkg = data.pop(f"{self.name}-rpos")
        btn_pkg = data.pop(f"{self.name}-btns")
        pos = np.reshape(pos_pkg["value"], (-1, 2))
        btns = np.reshape(btn_pkg["value"], (-1, 4))
        names = data_names.pop(f"{self.name}-names")["value"]
        pos = {
            f"{self.name}-pos": Reading(
                timestamp=pos_pkg["timestamp"],
                value=OrbitModel(
                    orbit=[
                        BPMReading(
                            name=str(name),
                            pos=BPMPosition(x=float(p[0]), y=float(p[1])),
                            btns=BPMButtons(*map(float, b)),
                        )
                        for name, p, b in zip(names, pos, btns)
                    ]
                ),
            )
        }
        data.update(pos)
        return jsons.dump(data)
