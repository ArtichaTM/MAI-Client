from typing import Any, TYPE_CHECKING
from queue import Queue

from mai.capnp.names import MAIControls, MAIGameState
from mai.capnp.data_classes import RunType, RunParameters

from .tactics.bases import BaseTactic
from .tactics.run_module import RunModule
from .tactics import trainings

if TYPE_CHECKING:
    from mai.capnp.data_classes import AdditionalContext
    from mai.ai.controller import NNController

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
                controls = MAIControls.new_message()
                controls.skip = True
                return controls
            self._current_tactic = self._tactics.get_nowait()
            self._current_tactic.prepare()
            if self._current_tactic.finished:
                self._current_tactic = None
                return
        controls = self._current_tactic.react(state, context)
        if self._current_tactic.finished:
            self._current_tactic = None
        return controls

    def add_reaction_tactic(self, tactic: BaseTactic) -> None:
        assert isinstance(tactic, BaseTactic)
        self._tactics.put_nowait(tactic)

    def clear_all_tactics(self) -> None:
        self._tactics = Queue()
        self._current_tactic = None

    def train(self, nnc: 'NNController', params: RunParameters) -> None:
        reaction = None
        match params.type:
            case RunType.CUSTOM_TRAINING:
                reaction = trainings.CustomTraining(nnc, params)
            case RunType.FREEPLAY:
                reaction = trainings.CustomTraining(nnc, params)
            case RunType.v11:
                pass
            case RunType.v22:
                pass
            case RunType.v33:
                pass
            case RunType.v44:
                pass
        if reaction is None:
            raise ValueError(
                f"Can't run module training "
                f"with type != `{RunType.CUSTOM_TRAINING.value}`. "
                "Add more modules for complete training or set "
                f"training type to {RunType.CUSTOM_TRAINING.value}"
            )

    def play(self, nnc: 'NNController', params: RunParameters) -> None:
        raise NotImplementedError()

    def pause(self) -> bool:
        assert self._paused_tactics is None
        if self._current_tactic is None:
            return False

        self._paused_tactics = Queue()
        self._paused_tactics.put(self._current_tactic)
        self._current_tactic = None
        while self._tactics.not_empty:
            self._paused_tactics.put(self._tactics.get(block=False))
        return True

    def resume(self) -> None:
        assert self._paused_tactics is not None
        self._current_tactic = self._paused_tactics.get()
        self._tactics = self._paused_tactics
        self._paused_tactics = None

    def stop(self) -> None:
        self.clear_all_tactics()
