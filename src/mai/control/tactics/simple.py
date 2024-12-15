from mai.capnp.names import MAIControls
from .bases import BaseTactic
from mai.capnp.data_classes import NormalControls
from mai.windows import WindowController


class SequencedCommands(BaseTactic):
    __slots__ = ('commands',)

    def __init__(self, *commands: NormalControls) -> None:
        super().__init__()
        self.commands = list(reversed(commands))

    def react(self, state, context):
        command = self.commands.pop()
        assert isinstance(command, NormalControls)
        if not len(self.commands):
            self.finished = True
        return command.toMAIControls()


class ButtonPress(BaseTactic):
    __slots__ = ('key',)
    key: int | None

    def __init__(self, key: int) -> None:
        super().__init__()
        assert isinstance(key, int)
        assert 1 <= key <= 254
        self.key = key

    def react(self, state, context) -> MAIControls:
        """
        Presses key and waits one exchange before reset is complete
        """
        win = WindowController._instance
        assert win is not None
        if self.key is None:
            self.finished = True
            return MAIControls()
        win.press_key(self.key)
        self.key = None
        return MAIControls()
