from dataclasses import asdict
from typing import Dict

from bluesky.protocols import Reading
from event_model import DataKey

from bact_device_models.devices.bpm_turn_by_turn_data import BPMTurnByTurnData
from ..raw.bpm import BPM as BPMR


class BPM(BPMR):
    async def describe(self) -> dict[str, DataKey]:
        d = await super().describe()
        x = d.pop(f"{self.name}-x")
        y = d.pop(f"{self.name}-y")
        d2 = {f"{self.name}-tbt": DataKey(source="", shape=[], dtype="array")}
        d.update(d2)
        return d

    async def read(self) -> dict[str, Reading]:
        data = await super().read()
        tbt = turn_by_turn_signals_to_data_model(
            data.pop(f"{self.name}-x"), data.pop(f"{self.name}-y"), self.name
        )
        data.update(tbt)
        return data


def turn_by_turn_signals_to_data_model(
    x: Reading, y: Reading, name: str
) -> Dict[str, Reading]:
    tx = x["timestamp"]
    ty = y["timestamp"]
    t = tx + (ty - tx) / 2.0
    dm = BPMTurnByTurnData(
        x=x["value"],
        y=y["value"],
        name=name,
        # Timestamps can not be added if these are datetimes
        timestamp=t,
    )
    return {f"{name}-tbt": Reading(timestamp=t, value=asdict(dm))}
