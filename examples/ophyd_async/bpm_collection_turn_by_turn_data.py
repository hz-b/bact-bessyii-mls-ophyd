import logging

logging.basicConfig(level=logging.WARNING)

import asyncio
import pprint
from typing import Dict, Union, Sequence, Literal

import jsons

from bact_bessyii_mls_ophyd.devices.pp.bpm_collection import BPMCollection
from bact_device_models.devices.bpm_turn_by_turn_data import (
    BPMTurnByTurnDataCollection,
    BPMTurnByTurnData, )
from bact_device_models.devices import data_window
from bact_bessyii_mls_ophyd.devices.pp.bpm import BPM

valid_keys = Literal["x", "y", "name", "timestamp"]


def enforce_basis_types(
    data: Dict[valid_keys, Union[Sequence[int], float, str]]
) -> Dict[str, Union[Sequence[int], float, str]]:
    """But ensure that the data is really that way ..."""
    data = data.copy()
    data["x"] = list(map(int, data.pop("x")))
    data["y"] = list(map(int, data.pop("y")))
    data["name"] = str(data.pop("name"))
    return data


async def main():
    col = BPMCollection(
        name="bpms",
        devices=[
            BPM(prefix="BPMZ41D1R:", name="bpmz41", select_slice=slice(0, 2000)),
            BPM(prefix="BPMZ42D1R:", name="bpmz42", select_slice=slice(0, 2000)),
        ],
        combine_suffixes=["tbt"],
    )
    await col.connect()
    await col.trigger()
    print("Describing")
    desc = await col.describe()
    print("->")
    pprint.pprint(desc)
    print("================")
    print("Reading")
    r = await col.read()
    print("->")
    print("================")
    (key,) = list(r)
    data = r[key]
    col = data["value"]
    # Todo: why do I need to enforce the basis types here?
    #       why it still sees numpy values
    dm1 = jsons.load(enforce_basis_types(col["col"][0]), BPMTurnByTurnData)
    dm = jsons.load(
        dict(col=[enforce_basis_types(bpm_d) for bpm_d in col["col"]]),
        BPMTurnByTurnDataCollection,
    )
    pprint.pprint(dm)
    print("================")


if __name__ == "__main__":
    asyncio.run(main())
