from typing import Annotated as A, Sequence

import numpy as np
from ophyd_async.core import SignalR, StandardReadable, Array1D
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice, PvSuffix


class Orbit(StandardReadable, EpicsDevice):
    # fmt:off
    names: A[ SignalR[ Sequence[str]       ], PvSuffix( "rdBpmNames" ), Format.CONFIG_SIGNAL   ]

    count: A[ SignalR[ int                 ], PvSuffix( "count"      ), Format.UNCACHED_SIGNAL ]

    rpos:  A[ SignalR[ Array1D[np.float64] ], PvSuffix( "rdPos"      ), Format.UNCACHED_SIGNAL ]
    btns:  A[ SignalR[ Array1D[np.float64] ], PvSuffix( "rdButtons"  ), Format.UNCACHED_SIGNAL ]
    # fmt:on


if __name__ == "__main__":
    import asyncio

    orbit = Orbit("ORBITCC:", name="orb")

    async def test():
        await orbit.connect()
        r = await orbit.read()
        print(r)

    asyncio.run(test())
