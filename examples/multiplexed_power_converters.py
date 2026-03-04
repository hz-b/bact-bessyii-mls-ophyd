import asyncio
import pprint
from typing import Any

from bact_bessyii_mls_ophyd.devices.utils.multiplexer_for_settable_devices import MultiplexerProxy, _ProxiedDeviceName
from bact_bessyii_mls_ophyd.devices.utils.power_converter import PowerConverter


async def test():
    # names or identifiers should be available from yellow pages
    _pc_names = [f"VS2P1T{sector}R:" for sector in range(1, 9)]
    _pc_names += [f"VS2P1D{sector}R:" for sector in range(1, 9)]
    _pc_names = [f"Anonym:{name}" for name in _pc_names][:3]

    # Now let's construct the power converters
    pcs = {
        name: PowerConverter(prefix, name=name, readback_suffix="rdbk", setpoint_suffix="set")
        for prefix, name in [(prefix, prefix.split(":")[1].lower()) for prefix in _pc_names]
    }

    pc_names = list(pcs.keys())
    sel = _ProxiedDeviceName(
        name=f"mux-sel",
        default_name="unknown",
        acceptable_names=list(pcs.keys()),
    )
    await sel.connect()
    r = await sel.read()
    d = await sel.describe()
    # print(d, r)

    # Now let's instantiate the multiplexer
    mux = MultiplexerProxy(name="mux", settable_devices=pcs, default_name=pc_names[0])
    stat = pcs[pc_names[0]].connect()
    r = await stat
    await mux.connect()
    d = await mux.describe()
    r = await mux.read()
    # pprint.pprint(d)
    # pprint.pprint(r)


    for pc_name in [pc_names[0], pc_names[-1]]:
        print(f"setting name: {pc_name}")
        await mux.sel.set(pc_name)
        await mux.proxy.set(-1e-3)

        r = await mux.read()
        pprint.pprint(r)

    # invalid_name = "not-a-name"
    # assert invalid_name not in pc_names
    # await mux.sel.set(invalid_name)

asyncio.run(test())
