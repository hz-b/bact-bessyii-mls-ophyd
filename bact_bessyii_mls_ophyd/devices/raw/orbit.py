from typing import Annotated as A, Sequence

import numpy as np
from ophyd_async.core import SignalR, SignalRW, StandardReadable, Array1D
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice, PvSuffix


class Orbit(EpicsDevice, StandardReadable):
    count: A[SignalR[int], PvSuffix("count"), Format.UNCACHED_SIGNAL]
    pos: A[SignalR[Array1D[np.float64]], PvSuffix("rdPos"), Format.UNCACHED_SIGNAL]
    btns: A[SignalR[Array1D[np.float64]], PvSuffix("rdButtons"), Format.UNCACHED_SIGNAL]
    names: A[SignalR[Sequence[str]], PvSuffix("rdBpmNames"), Format.UNCACHED_SIGNAL]

    async def read(self):
        r = await super().read()
        # Add processing as already done in ophyd ...
        # check if matching describe is necessary
        return r


if __name__ == "__main__":
    import asyncio

    orbit = Orbit("ORBITCC:", name="orb")


    async def test():
        await orbit.connect()
        r = await orbit.read()
        print(r)


    asyncio.run(test())
