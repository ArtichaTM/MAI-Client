from functools import partial
from time import sleep

from mai.functions import popup
from .interface import getAddressPort, runMainInterface
from .capnp.client import CapnPClient
from .capnp.exchanger import Exchanger
from .capnp.names import MAIControls

def check_socket(address: str, port: int, client: CapnPClient) -> bool:
    try:
        client.connect(address, port)
    except ConnectionRefusedError as e:
        popup(
            'Error',
            f"Error {e.args[0]}: {e.args[1]}"
        )
        return False
    print('Connected!')
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


if __name__ == '__main__':
    main()
