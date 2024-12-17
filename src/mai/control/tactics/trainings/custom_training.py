from typing import TYPE_CHECKING
from time import perf_counter

from mai.functions import popup
from mai.windows import WindowController
from mai.settings import WinButtons
from .base import ModuleTrainingTactic

if TYPE_CHECKING:
    from mai.capnp.data_classes import MAIGameState, AdditionalContext

class CustomTraining(ModuleTrainingTactic):
    __slots__ = ()

    def react_gen(self):
        state, context = yield
        state: 'MAIGameState'
        context: 'AdditionalContext'

        if context is None:
            print(context)
            popup("Error during context evaluation", (
                "For some reason no context is available. That usually\n"
                "happens when user starting training immediately after\n"
                "entering the match. Reloading in custom training should help"
            ))
            return

        if not context.latest_message != 'kickoffTimerStarted':
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
        state, context = yield

        while True:
            print('Starting!')
            start_time = perf_counter()
            while not self.is_restarting(
                state,
                context,
                time_since_start=start_time
            ):
                controls = self._nnc.exchange(state, context)
                yield controls.toNormalControls().toMAIControls()

            self._nnc.

            print('Restart!')
            keys.press_key(WinButtons.RESTART_TRAINING)

