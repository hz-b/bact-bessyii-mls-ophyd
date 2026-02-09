import logging
logging.basicConfig(level=logging.WARNING)

import asyncio
import datetime
import itertools

from bluesky import RunEngine
import bluesky.plans as bp
from databroker import catalog
import jsons


from bact_bessyii_mls_ophyd.devices.pp.bpm_collection import BPMCollection
from bact_device_models.devices.bpm_turn_by_turn_data import (
    BPMTurnByTurnDataCollection,
)
from bact_bessyii_mls_ophyd.devices.pp.bpm import BPM

logger = logging.getLogger("bact-bessyii-mls-ophyd")


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


def measure():
    bpm_names = create_bessyii_bpm_names()
    col = BPMCollection(
        name="col",
        devices=[
            # 10 is the limit the server can store
            # in 16 MB if all turns are stored
            # when limiting to 7000 turns then I can store all
            # 8000 turns are slightly above this limit
            BPM(prefix=f"{bpm_name}:", name=bpm_name, select_slice=slice(7000))
            for bpm_name in bpm_names
        ],
        combine_suffixes=["tbt"],
    )

    async def connect():
        await col.connect(timeout=10)
    asyncio.run(connect())

    RE = RunEngine({"target": "test_run"})
    db = catalog["heavy_local"]
    RE.subscribe(db.v1.insert)
    (uuid,) = RE(bp.count([col], 10))
    print(f"run engine measured {uuid=}")
    return uuid




def retrieve(uuid):
    start = datetime.datetime.now()
    db = catalog["heavy_local"]
    data = db[uuid]
    data_accessed = datetime.datetime.now()
    run = data.primary.read()
    run_read = datetime.datetime.now()
    bpm_data = [
        jsons.load(one_reading, BPMTurnByTurnDataCollection)
        for one_reading in run["col-col"].values
    ]
    data_converted = datetime.datetime.now()
    logger.warning(
        "Time required for:"
        " access to data %s,"
        " run read %s,"
        " bpm data converted %s",
        data_accessed - start,
        run_read - data_accessed,
        data_converted - run_read,
    )
    return bpm_data


def main():
    # uuid = measure()
    # print(f"measured to {uuid=}")
    # time.sleep(10)
    # return
    # uuid = "204c4de2"
    # uuid = "edc81bcc"
    # uuid = "8d78d445"
    # uuid = "e4b5e543"
    # uuid = "febc7082"
    # uuid = -            # when limiting to 7000 turns then I can store all
    uuid = "ce61ec5b"
    uuid = "f77c1dd1"
    print(f"loading {uuid=}")
    bpm_data = retrieve(uuid)
    bpm_data


if __name__ == "__main__":
    main()
