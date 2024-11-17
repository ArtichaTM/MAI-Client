from functools import partial
import socket

from mai.functions import popup
from .interface import getAddressPort, runMainInterface
from .capnp.client import CapnPClient
from .capnp.exchanger import Exchanger

def check_socket(address: str, port: int, client: CapnPClient, no_popup: bool = False) -> bool:
    try:
        client.connect(address, port)
    except (
        ConnectionRefusedError,
        socket.gaierror
    ) as e:
        if not no_popup:
            popup(
                'Error',
                f"Error {e.args[0]}: {e.args[1]}"
            )
        return False
    return True


def main():
    client = CapnPClient()
    checker = partial(check_socket, client=client)
    with client:
        client: CapnPClient
        address, port = getAddressPort(checker)
        if address is None:
            return
        exchanger = Exchanger(client)
        exchanger.run_forever_threaded()
        runMainInterface()
        exchanger.join()

if __name__ == '__main__':
    main()
