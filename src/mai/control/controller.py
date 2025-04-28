from typing import Any, TYPE_CHECKING
from queue import Queue

from mai.capnp.names import MAIControls, MAIGameState
from mai.capnp.data_classes import RunType, RunParameters

from mai.functions import create_dummy_controls
from .tactics.bases import BaseTactic
from .tactics import trainings

if TYPE_CHECKING:
    from mai.capnp.data_classes import AdditionalContext


class MainController:
    __slots__ = (
        '_tactics', '_current_tactic', '_paused_tactics',
    )
    _current_tactic: BaseTactic | None
    _tactics: Queue[BaseTactic]
    _paused_tactics: Queue[BaseTactic] | None

    def __init__(self) -> None:
        self._tactics = Queue()
        self._current_tactic = None
        self._paused_tactics = None

    def react(self, state: MAIGameState, context: 'AdditionalContext') -> MAIControls:
        if self._current_tactic is None:
            if self._tactics.empty():
                return create_dummy_controls()
            self._current_tactic = self._tactics.get_nowait()
            assert isinstance(self._current_tactic, BaseTactic), self._current_tactic
            self._current_tactic.prepare()
            if self._current_tactic.finished:
                self._current_tactic = None
                return create_dummy_controls()
        assert isinstance(self._current_tactic, BaseTactic), self._current_tactic
        controls = self._current_tactic.react(state, context)
        if self._current_tactic is not None:
            assert isinstance(self._current_tactic, BaseTactic), self._current_tactic
            if self._current_tactic.finished:
                self._current_tactic = None
        return controls

    def add_reaction_tactic(self, tactic: BaseTactic) -> None:
        assert isinstance(tactic, BaseTactic)
        self._tactics.put_nowait(tactic)

    def clear_all_tactics(self) -> None:
        if self._current_tactic is not None:
            self._current_tactic.close()
            self._current_tactic = None
        self._tactics = Queue()

    def is_training(self) -> bool:
        return isinstance(self._current_tactic, trainings.BaseTrainingTactic)

    def train(self, params: RunParameters) -> None:
        tactic = None
        match params.type:
            case RunType.CUSTOM_TRAINING:
                if len(params.modules) == 1:
                    tactic = trainings.CustomTraining(params)
                else:
                    assert len(params.modules) > 1
                    tactic = trainings.MAICustomTraining(params)
            case RunType.FREEPLAY:
                assert len(params.modules) == 1
                tactic = trainings.FreeplayTraining(params)
            case RunType.v11:
                raise NotImplementedError()
            case RunType.v22:
                raise NotImplementedError()
            case RunType.v33:
                raise NotImplementedError()
            case RunType.v44:
                raise NotImplementedError()

        if tactic is None:
            raise ValueError(
                f"Can't run module training "
                f"with type != `{RunType.CUSTOM_TRAINING.value}`. "
                "Add more modules for complete training or set "
                f"training type to {RunType.CUSTOM_TRAINING.value}"
            )
        self.add_reaction_tactic(tactic)

    def play(self, params: RunParameters) -> None:
        raise NotImplementedError()

    def pause(self) -> bool:
        assert self._paused_tactics is None
        if self._current_tactic is None:
            return False

        self._paused_tactics = Queue()
        self._paused_tactics.put(self._current_tactic)
        self._current_tactic = None
        while not self._tactics.empty():
            self._paused_tactics.put(self._tactics.get(block=False))
        return True

    def resume(self) -> None:
        assert self._paused_tactics is not None
        self._current_tactic = self._paused_tactics.get()
        self._tactics = self._paused_tactics
        self._paused_tactics = None

    def stop(self) -> None:
        self.clear_all_tactics()
