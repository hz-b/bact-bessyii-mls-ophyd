import logging

logging.basicConfig(level=logging.WARNING)

import asyncio
import pprint

from bact_bessyii_mls_ophyd.devices.pp.bpm import BPM


async def main():
    bpm = BPM(prefix="BPMZ41D1R:", name="bpm")
    await bpm.connect()
    await bpm.trigger()
    r2 = await bpm.read()
    pprint.pprint(r2)

    await bpm.trigger()
    r1 = await bpm.read()

    await bpm.trigger()
    r0 = await bpm.read()

    (key,) = list(r1)
    print(key)
    t2 = r2[key]["timestamp"]
    t1 = r1[key]["timestamp"]
    t0 = r0[key]["timestamp"]
    print(f"Time difference {t1-t2=:.3f} {t0-t2=:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
