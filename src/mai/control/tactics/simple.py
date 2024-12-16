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
    __slots__ = ('key', 'skip_frame')
    key: int | None

    def __init__(self, key: int, skip_frame: bool = True) -> None:
        super().__init__()
        assert isinstance(key, int)
        assert 1 <= key <= 254
        assert isinstance(skip_frame, bool)
        self.key = key
        self.skip_frame = skip_frame

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
        if self.skip_frame:
            self.key = None
        else:
            self.finished = True
        return MAIControls()
