from mai.settings import Settings
from .address_port import AddressPort
from .main_interface import MainInterface

def main():
    init_window = AddressPort()
    init_window.run()
    address, port = init_window.address, init_window.port
    if address is None:
        return
    assert isinstance(address, str)
    assert isinstance(port, int)
    Settings.server_address = (address, port)
    main_interface = MainInterface()
    main_interface.run()
