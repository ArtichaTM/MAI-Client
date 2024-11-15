from .address_port import AddressPort
from mai.settings import Settings

def main():
    init_window = AddressPort()
    init_window.run()
    address, port = init_window.address, init_window.port
    if address is None:
        return
    assert isinstance(address, str)
    assert isinstance(port, int)
    Settings.server_address = (address, port)
