"""A collection of power converters exported as if these were multiplexed

Todo:
    Review if the multiplex itself should be contained in this file too
    Or a helper function to generate it ...
"""
import logging
import time
from typing import Dict,  Union

from bluesky.protocols import Movable, Stageable, Reading
from event_model import DataKey
from ophyd.status import Status
from ophyd_async.core import StandardReadable
logger = logging.getLogger("bact-bessyii-ophyd")


def rename_dict_keys(d: Dict[str, Reading], src_prefix: str, tgt_prefix: str) -> Dict[str, Reading]:
    logger.debug('Renaming prefix from "%s" to "%s"', src_prefix, tgt_prefix)

    def rename_key(key):
        tmp, r = key.split(src_prefix)
        assert tmp == ""
        new_key = tgt_prefix + r
        return new_key

    return {rename_key(key): item for key, item in d.items()}


class _ProxiedDeviceName(Movable, StandardReadable, Stageable):
    """Select a power converter and mangle its data ...

    ProxiedDeviceName

    This part lets set a value. its sole purpose is to provide a
    name that can be set.

    Expected to be used together with PowerConverterCollectionAsMuxer

    Todo:
        Investigate how to
    """
    def __init__(self, *args, default_name: str = "Unknown", **kwargs):
        self.proxied_name = default_name
        self.timestamp = time.time()
        super().__init__(*args, **kwargs)

    def describe(self) -> dict[str, DataKey]:
        return {f"{self.name}-proxied_name": DataKey(source="", dtype="string", shape=[])}

    def read(self) -> dict[str, Reading]:
        return {f"{self.name}-proxied_name": Reading(value=self.proxied_name, timestamp=self.timestamp)}

    def set(self, name: str) -> Status:
        self.timestamp = time.time()
        self.proxied_name = name
        return Status(success=True, done=True)

    def get_name(self) -> str:
        return self.proxied_name


class _MultiplexerItemProxy(Movable, StandardReadable, Stageable):
    def __init__(self, *args, name: str="",
                 settable_devices: Dict[str, Union[Movable, StandardReadable, Stageable]],
                 selected: _ProxiedDeviceName,
                 **kwargs):
        self.settable_devices = settable_devices
        self.proxied_dev_name = selected
        self.proxied_device = None
        super().__init__(name=name)
        self.proxied_device = self.settable_devices[self.proxied_dev_name.get_name()]

    def _rename_keys(self, d):
        return rename_dict_keys(d, self.proxied_device.name, self.name)

    async def describe_configuration(self):
        d = await super().describe_configuration()
        d.update(self._rename_keys(d))
        return d

    async def read_configuration(self):
        d = await super().read_configuration()
        d.update(self._rename_keys(d))
        return d

    async def describe(self):
        d = await super().describe()
        d.update(self._rename_keys(d))
        return d

    async def read(self):
        d = await super().read()
        d.update(self._rename_keys(d))
        return d

    async def stop(self):
        pass


class MultiplexerProxy(StandardReadable, Stageable):
    def __init__(self, *args, name: str = "", settable_devices: Dict[str, Union[Movable, StandardReadable, Stageable]], **kwargs):
        with self.add_children_as_readables():
            # referenced here to be accessible
            self.sel = _ProxiedDeviceName(f"{name:}-sel")
            self.proxy = _MultiplexerItemProxy(f"{name:}-proxy", selected=self.sel, settable_devices=settable_devices)
        super().__init__(*args, **kwargs)

    def stop(self, success=False):
        return await self.proxy.stop()



__all__ = [MultiplexerProxy]
