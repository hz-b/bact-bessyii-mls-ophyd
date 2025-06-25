from ophyd_async.epics.core import EpicsDevice, PVSuffix, SignalR, SignalRW

from .pv_positioner_like_utils import _SettableControllingDifference


class MLSPowerConverter(EpicsDevice, _SettableControllingDifference):
    """
    Todo:
        move to MLS specific part
    """

    # fmt:off
    setpoint:  A[ SignalRW [ float ], PvSuffix( "setCur"      ), Format.UNCACHED_SIGNAL ]
    readback:  A[ SignalR  [ float ], PvSuffix( "rdCur"       ), Format.HINTED_SIGNAL   ]
    units:     A[ SignalR  [ str   ], PvSuffix( "rdCur.EGU"   ), Format.CONFIG_SIGNAL   ]
    precision: A[ SignalR  [ int   ], PvSuffix( "setCur.PREC" ), Format.CONFIG_SIGNAL   ]
    # fmt:on
