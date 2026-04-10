"""Access to a single Beam position monitor

If you need all data synchronously, please consider using the
orbit object
"""
from typing import Annotated as A

import numpy as np
from bluesky.protocols import Triggerable
from ophyd_async.core import (
    Array1D,
    AsyncStatus,
    StandardReadable,
    StandardReadableFormat as Format,
    SignalR,
)
from ophyd_async.epics.core import EpicsDevice, PvSuffix
from ..utils.utils import wait_for_new_value


class BPM(EpicsDevice, StandardReadable, Triggerable):
    # fmt:off
    x:   A[ SignalR[ Array1D[np.int32] ], PvSuffix("signals:tdp_synth.X"), Format.UNCACHED_SIGNAL ]
    y:   A[ SignalR[ Array1D[np.int32] ], PvSuffix("signals:tdp_synth.Y"), Format.UNCACHED_SIGNAL ]
    sum: A[ SignalR[ Array1D[np.int32] ], PvSuffix("signals:tdp_synth.Sum"), Format.UNCACHED_SIGNAL ]
    # fmt:on

    @AsyncStatus.wrap
    async def trigger(self):
        super().trigger()
        await wait_for_new_value(self.y)


if __name__ == "__main__":
    import asyncio
    import pprint

    async def main():
        bpm = BPM(prefix="BPMZ41D1R:", name="bpm")
        await bpm.connect()
        await bpm.trigger()
        r = await bpm.read()
        pprint.pprint(r)

    asyncio.run(main())
