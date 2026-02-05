from dataclasses import asdict
import time
from typing import Dict, Sequence

from bluesky.protocols import Reading
from event_model import DataKey

from bact_bessyii_mls_ophyd.devices.pp.device_collection import DeviceCollection
from bact_device_models.devices.bpm_turn_by_turn_data import BPMTurnByTurnData, BPMTurnByTurnDataCollection


class BPMCollection(DeviceCollection):
    def __init__(self, *args, combine_suffixes : Sequence[str],**kwargs):
        super().__init__(*args, **kwargs)
        self.combine_suffixes = combine_suffixes

    async def describe(self) -> dict[str, DataKey]:
        d = await super().describe()
        for dev in self.devices:
            for suffix in self.combine_suffixes:
                d.pop(f"{dev.name}-{suffix}")
        d2 = {f"{self.name}-col": DataKey(source="", shape=[], dtype="array")}
        d.update(d2)
        return d

    async def read(self) -> dict[str, Reading]:
        data = await super().read()
        d= {
            f"{self.name}-col" : dict(
                    value=asdict(combine_bpm_data_models(data)),
                    timestamp=time.time()
            )
        }
        return d


def combine_bpm_data_models(data: Dict[str, Reading]) ->  BPMTurnByTurnDataCollection:
    col = []
    suffix = None
    for k, v in data.items():
        if suffix is None:
            name, sep, suffix = k.rpartition("-")
            assert sep, f"Could not find suffix in key {k}"

        assert k.endswith(f"-{suffix}"), f"Data entry {k} does not end with '-{suffix}'"
        assert v["value"]["name"] == k[: -(len(suffix) + 1)],(
            f'BPM Data Collection name {v["value"]["name"]}"'
            f' does not match dict key {k} up to suffix {suffix}'
        )
        col.append(v["value"])
    return BPMTurnByTurnDataCollection(col=col)
