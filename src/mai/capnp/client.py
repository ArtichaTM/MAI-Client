import socket
from typing import Optional

from .names import MAIGameState, MAIControls


__all__ = ('CapnPClient',)

class CapnPClient:
    __slots__ = (
        'socket',
    )

    def __init__(self) -> None:
        self.socket: socket.socket | None = None

    def __enter__(self) -> 'CapnPClient':
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def receive(self) -> Optional[MAIGameState]:
        assert isinstance(self.socket, socket.socket)
        b = self.socket.recv(1024)
        if not b:
            return None
        with MAIGameState.from_bytes(b) as data:
            return data

    def send(self, controls: MAIControls) -> None:
        self.socket.sendall(controls.to_bytes())  # type: ignore

    def open(self) -> None:
        assert self.socket is None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def close(self) -> None:
        assert isinstance(self.socket , socket.socket)
        self.socket.close()
        self.socket = None

    def connect(self, address: str, port: int):
        assert isinstance(self.socket, socket.socket), 'Initialize client with open()'
        self.socket.connect((address, port))
