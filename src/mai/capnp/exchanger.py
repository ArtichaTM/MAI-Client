from typing import Callable, Optional
from time import sleep
from threading import Thread

from .names import MAIControls, MAIGameState
from .client import CapnPClient

__all__ = ('Exchanger', 'register_for_exchange')


class Exchanger:
    __slots__ = (
        '_listener', '_client', '_thread', '_exchanges_done',
        'sleep_time', 'stop',
    )
    _instance: Optional['Exchanger'] = None
    def __init__(self, client: CapnPClient) -> None:
        assert self._instance is None
        type(self)._instance = self
        self._client = client
        self._listener = None
        self._thread: Thread | None = None
        self.sleep_time: float = 0.3
        self.stop = False
        self._exchanges_done = 1

    @staticmethod
    def create_dummy_controls() -> MAIControls:
        message = MAIControls.new_message()
        message.skip = True
        return message

    def exchange(self, state: MAIGameState) -> MAIControls:
        if self._listener is None:
            return self.create_dummy_controls()
        controls = self._listener(state)
        if controls is None:
            return self.create_dummy_controls()
        return controls

    @classmethod
    def register_for_exchange(
        cls,
        function: Callable[[MAIGameState], Optional[MAIControls]]
    ) -> None:
        assert callable(function)
        assert cls._instance is not None
        assert cls._instance._listener is None, f"Listener already exists"
        cls._listener = function

    def _run(self) -> None:
        while not self.stop:
            if self.stop: return
            state = self._client.receive()
            if self.stop: return
            if state is None:
                print('Received None, closing')
                return
            controls = self.exchange(state)
            if self.stop: return
            sleep(self.sleep_time)
            if self.stop: return
            self._client.send(controls)
            self._exchanges_done += 1

    def run_forever_threaded(self) -> None:
        assert self._thread is None
        self._thread = Thread(
            target=self._run,
            name='Exchanger thread',
            daemon=True
        )
        self._thread.start()

    def join(self) -> None:
        self._client.socket = None
        self.stop = True
        assert self._thread is not None
        self._thread.join(self.sleep_time+0.1)


def register_for_exchange(function: Callable[[MAIGameState], MAIControls]):
    Exchanger.register_for_exchange(function)
