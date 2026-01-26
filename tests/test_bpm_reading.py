import asyncio
import pprint

from  bact_bessyii_mls_ophyd.devices.pp.orbit import PPOrbit


async def main():
    orb = PPOrbit(prefix="ORBITCC:", name="orb");

    await orb.connect()
    r = await orb.read()
    pprint.pprint(r)
    

if __name__ == "__main__":
    asyncio.run(main())
