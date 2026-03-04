import asyncio
import pprint

from bact_bessyii_mls_ophyd.devices.pp.orbit import PPOrbit


async def main():
    orbit = PPOrbit("ORBITCC:", name="orb")
    await orbit.connect()
    cfg = await orbit.read_configuration()
    print("Configuration")
    pprint.pprint(cfg)
    print("Data description")
    pprint.pprint(await orbit.describe())
    data= await orbit.read()
    print("Data")
    pprint.pprint(data)

if __name__ == "__main__":
    asyncio.run(main())