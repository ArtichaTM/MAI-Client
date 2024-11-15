from typing import Callable

from mai.settings import Settings
from .address_port import AddressPort
from .main_interface import MainInterface


__all__ = ('getAddressPort',)


def getAddressPort(checker: Callable[[str, int], bool]) -> tuple[str, int] | tuple[None, None]:
    while True:
        if Settings.server_address is None:
            init_window = AddressPort()
        else:
            init_window = AddressPort(*Settings.server_address)
        init_window.run()
        address, port = init_window.address, init_window.port
        if address is None:
            assert port is None
            return (address, port)
        assert isinstance(address, str)
        assert isinstance(port, int)
        output = checker(address, port)
        if output:
            return (address, port)

def run_main_interface(broker) -> None:
    interface = MainInterface()
    interface.run(broker)
