from typing import Annotated as A

from ophyd_async.core import SignalR, StandardReadable
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice, PvSuffix, epics_signal_r
from ophyd_async.core import Table

class Orbit(StandardReadable):
    # fmt:on
    # data: A[ SignalR [ Table ], PvSuffix( ":rdBPM"      ), Format.UNCACHED_SIGNAL ]

     def __init__(self, prefix: str, *, name: str):
        with self.add_children_as_readables():
            self.data = epics_signal_r(Table, f"{prefix}:rdBPM")
        super().__init__(name=name)


if __name__ == "__main__":
    import asyncio

    # that's the advantage of the table ... just read what you get
    # so could be sufficient to just read it in this manner
    # but don't forget it to give it a name ... otherwise it
    # ·uses an empty string
    orbit_pv = epics_signal_r(Table, "pva://ORBITCC:rdBPM", name="orbit")

    orbit = Orbit("pva://ORBITCC", name="orb")

    async def test():
        await asyncio.gather(orbit.connect(), orbit_pv.connect())
        r = await orbit.read()
        table = r["orb-data"]["value"]
        print(table)

        r = await orbit_pv.read()
        table = r["orbit"]["value"]
        print(table)
    asyncio.run(test())
