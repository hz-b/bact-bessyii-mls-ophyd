"""Read devices together as collection that return data as models
"""
import asyncio
from typing import Sequence, Union, Dict, Iterable, Awaitable, TypeVar, Set

from bluesky.protocols import Triggerable, Status, Reading
from event_model import DataKey
from ophyd_async.core import AsyncStatus, StandardReadable


class DeviceCollection(StandardReadable, Triggerable):
    def __init__(self, name, devices: Sequence[Union[StandardReadable, Triggerable]]):
        super().__init__(name=name)
        self.devices = devices

    @AsyncStatus.wrap
    async def connect(self, *args, **kwargs) -> Status:
        await asyncio.gather(*[dev.connect(*args, **kwargs) for dev in self.devices])
        return

    @AsyncStatus.wrap
    async def trigger(self) -> Status:
        await asyncio.gather(*[dev.trigger() for dev in self.devices])

    # @AsyncStatus.wrap
    async def describe(self) -> dict[str, DataKey]:
        rs = await super().describe()
        # The dict could be updated as it comes in
        # here all only done afterward for code sip
        # r  = await merge_gathered_dicts(
        #     [dev.describe() for dev in self.devices]
        # )
        r, collided = await merge_gathered_dictionaries_report_collisions(
            [dev.describe() for dev in self.devices]
        )
        if collided:
            self.log.warning(
                "%s: describe found collision for the following keys %s",
                self.name,
                collided,
            )
        return r

    # @AsyncStatus.wrap
    async def read(self) -> dict[str, Reading]:
        # The dict could be updated as it comes in
        # here all only done afterward for code sip
        # rs = await super().read()
        r, collided = await merge_gathered_dictionaries_report_collisions(
            [dev.read() for dev in self.devices]
        )
        if collided:
            self.log.warning(
                "%s: describe found collision for the following keys %s",
                self.name,
                collided,
            )
        return r


V = TypeVar("V")


async def merge_gathered_dictionaries_report_collisions(
    coros: Iterable[Awaitable[Dict[str, V]]]
) -> tuple[Dict[str, V], Set[str]]:
    """merge and keep report on collisions"""
    combined = {}
    collided = set()

    for fut in asyncio.as_completed(coros):
        for k, v in (await fut).items():
            # Just a safety net ... normally there should not be any
            # collisions
            if k in combined and k not in collided:
                collided.add(k)
            combined[k] = v

    return combined, collided
