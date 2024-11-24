from typing import Callable

from mai.settings import Settings
from .host_port import HostPort
from .main_interface import MainInterface


__all__ = ('getAddressPort', 'runMainInterface')


def getAddressPort(checker: Callable[[str, int], bool]) -> tuple[str, int] | tuple[None, None]:
    output = checker(*Settings.server_address, no_popup=True)
    if output:
        return Settings.server_address
    while True:
        init_window = HostPort()
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


def runMainInterface() -> int:
    interface = MainInterface()
    return interface.run()
