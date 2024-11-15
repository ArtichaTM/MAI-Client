from typing import Type
import socket

import capnp

from mai.settings import Settings
from .stub.fromRL import Data
from .stub.toRL import Controls


__all__ = ('CapnPBroker',)

class CapnPBroker:
    __slots__ = (
        'data_type',
        'controls_type',
        'socket'
    )

    def __init__(self) -> None:
        self.data_type: Type[Data] = capnp.load(str(  # type: ignore
            Settings.path_to_capnp_schemes.joinpath('fromRL.capnp')
        )).Data
        self.controls_type: Type[Controls] = capnp.load(str(  # type: ignore
            Settings.path_to_capnp_schemes.joinpath('toRL.capnp')
        )).Controls
        self.socket: socket.socket | None = None

    def __enter__(self) -> 'CapnPBroker':
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _receive(self) -> Data:
        assert isinstance(self.socket, socket.socket)
        print(dir(self.data_type))
        b = self.socket.recv(1024)
        data = self.data_type.from_bytes_packed(b)
        return data

    def _send(self, controls: Controls) -> None:
        self.socket.sendall(controls.to_bytes())  # type: ignore

    def exchange(self, controls: Controls) -> Data:
        self._send(controls)
        return self._receive()

    def open(self) -> None:
        assert self.socket is None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def close(self) -> None:
        assert isinstance(self.socket , socket.socket)
        self.socket.close()

    def connect(self, address: str, port: int):
        assert isinstance(self.socket, socket.socket)
        self.socket.connect((address, port))
