from functools import partial
from threading import Thread
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


def load_torch() -> None:
    from time import sleep
    sleep(0.3)
    import torch
    pass


def main():
    from time import perf_counter
    torch_load_thread = Thread(target=load_torch, name='Torch preload')
    torch_load_thread.start()
    while True:
        address, port = None, None
        client = CapnPClient()
        checker = partial(check_socket, client=client)
        exchanger = Exchanger(client)
        with client:
            address, port = getAddressPort(checker)
            if address is None:
                return
            exchanger.run_forever_threaded()
            if torch_load_thread is not None:
                torch_load_thread.join()
                torch_load_thread = None
            exit_code = runMainInterface()
            exchanger.join()
            if exit_code == 0:
                return

if __name__ == '__main__':
    main()
