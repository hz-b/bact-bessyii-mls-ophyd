import asyncio
import pprint

import jsons

from databroker import catalog
from bluesky import RunEngine
import bluesky.plans as bp

from bact_bessyii_mls_ophyd.devices.pp.orbit import Orbit
# from bact_device_models.devices.orbit import BPMReading, Orbit, BPMButtons, BPMPosition


async def measure() -> str:

    orb = Orbit("pva://ORBITCC", name="orbit")
    await orb.connect()

    RE=RunEngine({"target": "test_run"})
    db = catalog["heavy_local"]
    RE.subscribe(db.v1.insert)
    uuid, = RE(bp.count([orb], 3))
    return uuid

async def retrieve(uuid: str):
    db = catalog["heavy_local"]
    run = db[uuid]
    data = run.primary.read()
    orbit_data = [jsons.load(datum) for datum in data["orbit-pos"].data]
    pprint.pprint(orbit_data, compact=True, width=120)


async def main():
    uuid = await measure()
    # print(uuid)
    # uid = "e1ab4831"
    await retrieve(uuid)


if __name__ == "__main__":
    asyncio.run(main())