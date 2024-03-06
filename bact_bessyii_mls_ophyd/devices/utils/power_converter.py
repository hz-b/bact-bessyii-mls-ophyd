from ophyd import (
    EpicsSignal,
    EpicsSignalRO,
    PVPositionerPC,
    Component as Cpt,
    Signal,
    Kind,
)
from ophyd.status import SubscriptionStatus

# from ..utils import signal_with_validation
from .reached_setpoint import ReachedSetpointEPS
import numpy as np

_t_super = PVPositionerPC
_t_super = ReachedSetpointEPS


class PowerConverter(_t_super):
    """A power converter abstraction

    Currently only a device checking that the set and read value corresponds

    Todo:
        Insist on an hyseteres is loop
        Proper accuracy settings

        How to handle differences between MLS and BESSY II

        BESSY II uses

          ':set' / 'rdbk
          ':setCur / :rdCur'

       MLS uses
    '
    """

    setpoint = Cpt(EpicsSignal, ":set")
    readback = Cpt(EpicsSignalRO, ":rdbk")
    setpoint = Cpt(EpicsSignal, ":setCur")
    readback = Cpt(EpicsSignalRO, ":rdCur")


class ResettingPowerConverter(PowerConverter):
    """ """

    #: reference value to store
    rv = Cpt(Signal, name="ref_val", value=np.nan)

    #: shall the component be set back
    set_back = Cpt(Signal, name="set_bak", value=False, kind=Kind.config)

    #: acceptable relative error
    eps_rel = Cpt(Signal, name="eps_rel", value=6e-2, kind=Kind.config)

    #: execution stopped with a difference of 0.7 %
    #: at a value of 0.13
    eps_abs = Cpt(Signal, name="eps_abs", value=1e-2, kind=Kind.config)

    #: steerer always set at least the value once.
    always_set = Cpt(Signal, name="always_set", value=True, kind=Kind.config)

    def __init__(self, *args, **kwargs):
        # 10 ms is way too short
        # let's go for rather half a second
        kwargs.setdefault("settle_time", 0.5)
        kwargs.setdefault("timeout", 20)
        super().__init__(*args, **kwargs)

    def setToStoredValue(self):
        if self.set_back.get():
            val = self.rv.get()
            stat = self.setpoint.set(val)
            # stat.wait(2)

    def stage(self):
        """ """
        stat = self.rv.set(self.setpoint.get())
        # stat.wait(1.0)
        return super().stage()

    def unstage(self):
        """

        Warning:
            If the call to super is not here proper plans will stop
            working at the second iteration
        """
        return super().unstage()

    def stop(self, success=False):
        self.setToStoredValue()
