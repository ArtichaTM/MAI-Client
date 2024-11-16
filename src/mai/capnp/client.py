import socket

from .names import GameState, Controls


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

    def receive(self) -> GameState:
        assert isinstance(self.socket, socket.socket)
        b = self.socket.recv(1024)
        print(b)
        data = GameState.from_bytes_packed(b)
        return data

    def send(self, controls: Controls) -> None:
        self.socket.sendall(controls.to_bytes())  # type: ignore

    def open(self) -> None:
        assert self.socket is None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def close(self) -> None:
        assert isinstance(self.socket , socket.socket)
        self.socket.close()

    def connect(self, address: str, port: int):
        assert isinstance(self.socket, socket.socket), 'Initialize client with open()'
        self.socket.connect((address, port))
