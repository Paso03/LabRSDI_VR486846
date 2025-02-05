import sys
sys.coinit_flags = 0

try:
    from bleak.backends.winrt.util import allow_sta

    allow_sta()
except ImportError:
    pass

import asyncio
from utils.utility import scan, find
from classes import Thingy52Client


async def main():
    my_thingy_addresses = ["FB:7B:DF:44:C3:1A"]

    discovered_devices = await scan()
    my_devices = find(discovered_devices, my_thingy_addresses)

    thingy52 = Thingy52Client.Thingy52Client(my_devices[0])
    await thingy52.connect()
    thingy52.save_to(str(input("Enter recording name: ")))
    await thingy52.receive_inertial_data()

if __name__ == '__main__':
    asyncio.run(main())