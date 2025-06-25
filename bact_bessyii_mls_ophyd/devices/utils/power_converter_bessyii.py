from .pv_positioner_like_utils import _SettableControllingDifference
from ophyd_async.epics.core import EpicsDevice, PVSuffix, SignalR, SignalRW


class BESSYIIPowerConverter(EpicsDevice, _SettableControllingDifference):
    """
    Todo:
        move to BESSY II specific part
    """

    # fmt:off
    setpoint:  A[ SignalRW [ float ], PvSuffix( "set"      ), Format.UNCACHED_SIGNAL ]
    readback:  A[ SignalR  [ float ], PvSuffix( "rdbk"     ), Format.HINTED_SIGNAL   ]
    units:     A[ SignalR  [ str   ], PvSuffix( "rdbk.EGU" ), Format.CONFIG_SIGNAL   ]
    precision: A[ SignalR  [ int   ], PvSuffix( "set.PREC" ), Format.CONFIG_SIGNAL   ]
    # fmt:on


if __name__ == "__main__":
    import asyncio

    async def test():
        from .pv_positioner_like_utils import PVPositionerIsClose

        pc = BESSYIIPowerConverter("VS2P1T6R:", name="pc")
        pc = PVPositionerIsClose(
            "VS2P1T6R:", readback_suffix="rdbk", setpoint_suffix="set", name="pc"
        )
        await pc.connect()
        r = await pc.read()
        print(r)
        await pc.set(0.0)

    asyncio.run(test())
