from .bases import BaseTactic
from mai.functions import create_dummy_controls
from mai.capnp.data_classes import NormalControls
from mai.windows import WindowController


class SequencedCommands(BaseTactic):
    __slots__ = ('start', 'commands')

    def __init__(self, *commands: NormalControls) -> None:
        super().__init__()
        self.commands = commands

    def react_gen(self):
        for command in self.commands:
            assert isinstance(command, NormalControls)
            yield command.toMAIControls()


class ButtonPress(BaseTactic):
    __slots__ = ('key', 'skip_frame')
    key: int
    skip_frame: bool

    def __init__(self, key: int, skip_frame: bool = True) -> None:
        super().__init__()
        assert isinstance(key, int)
        assert 1 <= key <= 254
        assert isinstance(skip_frame, bool)
        self.key = key
        self.skip_frame = skip_frame

    def react_gen(self):
        """
        Presses key and waits one exchange before reset is complete
        """
        win = WindowController._instance
        assert win is not None
        win.press_key(self.key)
        if self.skip_frame:
            yield
