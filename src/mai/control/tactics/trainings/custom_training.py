from typing import TYPE_CHECKING

from .base import ModuleTrainingTactic
from mai.functions import popup
from mai.windows import WindowController
from mai.settings import WinButtons

if TYPE_CHECKING:
    from mai.capnp.data_classes import MAIGameState, AdditionalContext

class CustomTraining(ModuleTrainingTactic):
    __slots__ = ()

    def react_gen(self):
        state, context = yield
        state: 'MAIGameState'
        context: 'AdditionalContext'

        if context is None:
            popup("Error during context evaluation", (
                "For some reason no context is available. That usually\n"
                "happens when user starting training immediately after\n"
                "entering the match. Reloading in custom training should help"
            ))
            return

        if not str(context.latest_message).startswith('kickoffTimer'):
            print(context.latest_message)
            popup("Error during training startup", (
                "Trying to run training without entering custom training. \n"
                "Before starting training in custom training, make sure u \n"
                "entered custom training without pressing anything"
            ))
            return

        keys = WindowController._instance
        assert keys is not None
        keys.update_window()
        keys.press_key(WinButtons.FORWARD)
