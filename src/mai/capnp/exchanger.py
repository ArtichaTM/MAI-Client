from typing import Callable, Optional
from time import sleep
from threading import Thread

from .names import MAIControls, MAIGameState
from .client import CapnPClient
from .data_classes import Vector, AdditionalContext

__all__ = ('Exchanger', 'register_for_exchange')


class Exchanger:
    __slots__ = (
        '_listener', '_client', '_thread', '_exchanges_done',
        'sleep_time', 'stop', '_context', 'magnitude_update_requested',
    )
    _instance: Optional['Exchanger'] = None
    _listener_type = Callable[[MAIGameState, AdditionalContext], Optional[MAIControls]]

    def __init__(self, client: CapnPClient) -> None:
        assert self._instance is None
        type(self)._instance = self
        self._client = client
        self._listener = None
        self._thread: Thread | None = None
        self.sleep_time: float = 0.3
        self.stop = False
        self._exchanges_done = 1
        self._context: AdditionalContext | None = None
        self.magnitude_update_requested = False

    @property
    def context(self) -> AdditionalContext | None:
        return self._context

    @staticmethod
    def create_dummy_controls() -> MAIControls:
        message = MAIControls.new_message()
        message.skip = True
        return message

    def update_magnitudes(self, state: MAIGameState) -> None:
        if self._context is None:
            print("Can't update magnitudes: context is None")
            return
        temp = Vector.from_mai(state.car.velocity).magnitude()
        self._context.magnitude_offsets['car']['v'] = temp
        temp = Vector.from_mai(state.car.angularVelocity).magnitude()
        self._context.magnitude_offsets['car']['av'] = temp
        temp = Vector.from_mai(state.ball.velocity).magnitude()
        self._context.magnitude_offsets['ball']['v'] = temp
        temp = Vector.from_mai(state.ball.angularVelocity).magnitude()
        self._context.magnitude_offsets['ball']['av'] = temp

    def _update_context(self, state: MAIGameState) -> None:
        if self.magnitude_update_requested:
            self.update_magnitudes(state)
            self.magnitude_update_requested = False
        if state.message == 'none':
            return
        elif state.message == 'gameExit':
            self._context = None
            return
        elif state.message == 'kickoffTimerEnded':
            if self._context is None:
                self._context = AdditionalContext()
            car_y = state.car.position.y
            if car_y < 0:
                self._context.team_multiplier = -1
            else:
                self._context.team_multiplier = 1
        elif state.message == 'kickoffTimerStarted':
            if self._context is None:
                self._context = AdditionalContext()
            self.update_magnitudes(state)

        if self._context is not None:
            self._context.latest_message = state.message

    def exchange(self, state: MAIGameState) -> MAIControls:
        self._update_context(state)
        if self._listener is None:
            return self.create_dummy_controls()
        controls = self._listener(state, self._context)
        if controls is None:
            return self.create_dummy_controls()
        return controls

    @classmethod
    def register_for_exchange(
        cls,
        function: _listener_type
    ) -> None:
        assert callable(function)
        assert cls._instance is not None
        assert cls._instance._listener is None, f"Listener already exists"
        cls._instance._listener = function

    def _run(self) -> None:
        while not self.stop:
            if self.stop: return
            state = self._client.receive()
            if self.stop: return
            if state is None: return
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

    def isAlive(self) -> bool:
        assert self._thread is not None
        return self._client.socket is not None

    def join(self) -> None:
        self._client.socket = None
        self.stop = True
        assert self._thread is not None
        self._thread.join(self.sleep_time+0.1)
        type(self)._instance = None


def register_for_exchange(function: Exchanger._listener_type):
    Exchanger.register_for_exchange(function)
