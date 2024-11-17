from queue import Queue

from mai.capnp.names import MAIControls, MAIGameState

from .tactics.bases import BaseTactic

class MainController:
    __slots__ = ('_tactics', '_current_tactic')
    _current_tactic: BaseTactic | None
    _tactics: Queue[BaseTactic]

    def __init__(self) -> None:
        self._tactics = Queue()
        self._current_tactic = None

    def react(self, state: MAIGameState) -> MAIControls:
        if self._current_tactic is None:
            if self._tactics.empty():
                controls = MAIControls.new_message()
                controls.skip = True
                return controls
            self._current_tactic = self._tactics.get_nowait()
        controls = self._current_tactic.react(state)
        if self._current_tactic.finished:
            self._current_tactic = None
        return controls

    def add_reaction_tactic(self, tactic: BaseTactic) -> None:
        assert isinstance(tactic, BaseTactic)
        self._tactics.put_nowait(tactic)

    def clear_all_tactics(self) -> None:
        self._tactics = Queue()
        self._current_tactic = None
