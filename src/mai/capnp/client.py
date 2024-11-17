from typing import Optional
import socket
from time import sleep, perf_counter

from .names import MAIGameState, MAIControls


__all__ = ('CapnPClient',)

class CapnPClient:
    __slots__ = (
        'socket', 'sleep_time',
    )

    def __init__(self) -> None:
        self.socket: socket.socket | None = None
        self.sleep_time = 0

    def __enter__(self) -> 'CapnPClient':
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def receive(self) -> Optional[MAIGameState]:
        assert isinstance(self.socket, socket.socket)
        b = None
        while self.socket:
            try:
                b = self.socket.recv(1024)
            except BlockingIOError:
                sleep(self.sleep_time)
            except ConnectionResetError as e:
                assert e.errno == 10054
                self.close()
                return None
            else:
                break
        assert b is not None
        if not b:
            return None
        with MAIGameState.from_bytes(b) as data:
            return data

    def send(self, controls: MAIControls) -> None:
        while self.socket:
            try:
                self.socket.sendall(controls.to_bytes())  # type: ignore
            except BlockingIOError:
                sleep(self.sleep_time)
            else:
                break

    def open(self) -> None:
        assert self.socket is None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setblocking(False)

    def close(self) -> None:
        if self.socket is None: return
        assert isinstance(self.socket , socket.socket)
        self.socket.close()
        self.socket = None

    def connect(self, address: str, port: int):
        """Connects client to the server
        :raises ConnectionRefusedError: When connection can't be done
        """
        assert isinstance(self.socket, socket.socket), 'Initialize client with open()'
        assert isinstance(address, str)
        assert isinstance(port, int)
        e = self.socket.connect_ex((address, port))
        e = self.socket.connect_ex((address, port))
        if e != 10056:
            raise ConnectionRefusedError(
                e,
                f"After NonBlocking got another exception.\n"
                "Probably no connection can be made"
            )
