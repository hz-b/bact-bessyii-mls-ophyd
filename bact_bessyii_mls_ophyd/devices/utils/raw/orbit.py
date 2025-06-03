from typing import Annotated as A

#from ophyd_async.core import SignalR, SignalRW, StandardReadable
#from ophyd_async.core import StandardReadableFormat as Format
#from ophyd_async.epics.core import EpicsDevice, PvSuffix

#class Orbit(EpicsDevice, StandardReadable):
#    pass

if __name__ == "__main__":
    import ophyd_async
    print(ophyd_async.__version__)
 #   orbit = Orbit("OrbitCC:")