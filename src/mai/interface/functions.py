from typing import Callable

from mai.settings import Settings
from .host_port import HostPort
from .main_interface import MainInterface


__all__ = ('getAddressPort', 'runMainInterface')


def getAddressPort(checker: Callable[[str, int], bool]) -> tuple[str, int] | tuple[None, None]:
    while True:
        if Settings.server_address is None:
            init_window = HostPort()
        else:
            init_window = HostPort(*Settings.server_address)
        init_window.run()
        host, port = init_window.host, init_window.port
        if host is None:
            assert port is None
            return (host, port)
        assert isinstance(host, str)
        assert isinstance(port, int)
        output = checker(host, port)
        if output:
            return (host, port)

def runMainInterface() -> None:
    interface = MainInterface()
    interface.run()
