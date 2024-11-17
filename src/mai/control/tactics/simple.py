from typing import Sequence

from .bases import BaseTactic
from ..data_classes import NormalControls



class SequencedCommands(BaseTactic):
    __slots__ = ('commands',)

    def __init__(self, *commands: NormalControls) -> None:
        super().__init__()
        self.commands = list(reversed(commands))

    def react(self, state):
        command = self.commands.pop()
        assert isinstance(command, NormalControls)
        if not len(self.commands):
            self.finished = True
        return command.toMAIControls()
