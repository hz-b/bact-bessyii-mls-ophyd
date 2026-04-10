import logging


logging.basicConfig(level=logging.WARNING)

import asyncio
import pprint

from bluesky import RunEngine
import bluesky.plans as bp
from databroker import catalog
import jsons

from bact_device_models.devices.bpm_turn_by_turn_data import BPMTurnByTurnData
from bact_bessyii_mls_ophyd.devices.pp.bpm import BPM


async def measure():
    bpm = BPM(prefix="BPMZ41D1R:", name="bpm", select_slice=slice(0, 2000))

    await bpm.connect()

    RE = RunEngine({"target": "test_run"})
    db = catalog["heavy"]
    RE.subscribe(db.v1.insert)
    uuid, = RE(bp.count([bpm], 3))
    return uuid


def retrieve(uuid):
    db = catalog["heavy"]
    data = db[uuid]
    run = data.primary.read()
    bpm_data = [jsons.load(one_reading, BPMTurnByTurnData) for one_reading in run["bpm-tbt"].values]
    pprint.pprint(bpm_data[0])


async def main():
    uuid = await measure()
    # uuid = '15f7c8e9'
    retrieve(uuid)


if __name__ == "__main__":
    asyncio.run(main())
