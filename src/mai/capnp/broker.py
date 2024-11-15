import socket

from mai.settings import Settings
from .names import GameState, Controls


__all__ = ('CapnPBroker',)

class CapnPBroker:
    __slots__ = (
        'socket',
    )

    def __init__(self) -> None:
        self.socket: socket.socket | None = None

    def __enter__(self) -> 'CapnPBroker':
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _receive(self) -> GameState:
        assert isinstance(self.socket, socket.socket)
        b = self.socket.recv(1024)
        data = GameState.from_bytes_packed(b)
        return data

    def _send(self, controls: Controls) -> None:
        self.socket.sendall(controls.to_bytes())  # type: ignore

    def exchange(self, controls: Controls) -> GameState:
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
