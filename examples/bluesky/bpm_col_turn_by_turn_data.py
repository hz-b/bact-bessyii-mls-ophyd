import itertools
import logging

from pymongo import timeout

from bact_bessyii_mls_ophyd.devices.pp.bpm_collection import BPMCollection
from bact_bessyii_mls_ophyd.devices.pp.device_collection import DeviceCollection

logging.basicConfig(level=logging.WARNING)

import asyncio
import pprint

from bluesky import RunEngine
import bluesky.plans as bp
from databroker import catalog
import jsons

from bact_device_models.devices.bpm_turn_by_turn_data import (
    BPMTurnByTurnData,
    BPMTurnByTurnDataCollection,
)
from bact_bessyii_mls_ophyd.devices.pp.bpm import BPM


def create_bessyii_bpm_names():
    """Here to minimise external dependenices"""
    bpm_names = [
        [
            [f"BPMZ{idx}{focusing_type}{sector}R" for idx in range(1, 7 + 1)]
            for focusing_type in ("D", "T")
        ]
        for sector in range(1, 8 + 1)
    ]
    bpm_names = list(itertools.chain(*itertools.chain(*bpm_names)))
    bpm_names.remove("BPMZ5D6R")
    return bpm_names


async def measure():
    bpm_names = create_bessyii_bpm_names()
    col = BPMCollection(
        name="col",
        devices=[
            # 10 is the limit the server can store
            # in 16 MB
            BPM(prefix=f"{bpm_name}:", name=bpm_name) for bpm_name in bpm_names[:10]
        ],
        combine_suffixes=["tbt"],
    )

    await col.connect(timeout=10)

    RE = RunEngine({"target": "test_run"})
    db = catalog["heavy_local"]
    RE.subscribe(db.v1.insert)
    (uuid,) = RE(bp.count([col], 2))
    print("run engine measured {uuid=}")
    return uuid


def retrieve(uuid):
    db = catalog["heavy_local"]
    data = db[uuid]
    run = data.primary.read()
    bpm_data = [
        jsons.load(one_reading, BPMTurnByTurnDataCollection)
        for one_reading in run["col-col"].values
    ]
    return bpm_data


async def main():
    uuid = await measure()
    print(f"measured to {uuid=}")
    # return
    # uuid = "204c4de2"
    # uuid = "edc81bcc"
    # uuid = "8d78d445"
    print(f"loading {uuid=}")
    bpm_data = retrieve(uuid)
    bpm_data


if __name__ == "__main__":
    asyncio.run(main())
