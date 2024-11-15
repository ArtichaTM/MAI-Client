from functools import partial

from mai.functions import popup
from .interface import getAddressPort
from .capnp.client import CapnPClient


def check_socket(address: str, port: int, broker: CapnPClient) -> bool:
    try:
        broker.connect(address, port)
    except ConnectionRefusedError as e:
        popup(
            'Error',
            f"Error {e.args[0]}: {e.args[1]}"
        )
        return False
    return True


def main():
    broker = CapnPClient()
    checker = partial(check_socket, broker=broker)
    with broker:
        broker: CapnPClient
        address, port = getAddressPort(checker)
        if address is None:
            return


if __name__ == '__main__':
    main()
