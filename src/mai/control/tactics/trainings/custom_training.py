from typing import TYPE_CHECKING

from .base import ModuleTrainingTactic
from mai.functions import popup

if TYPE_CHECKING:
    from mai.capnp.data_classes import MAIGameState, AdditionalContext

class CustomTraining(ModuleTrainingTactic):
    __slots__ = ()

    def react_gen(self):
        state, context = yield
        state: 'MAIGameState'
        context: 'AdditionalContext'

        if context.latest_message != 'kickoffTimerEnded':
            popup("Error during training startup", (
                "Trying to run training without entering custom training. "
                "Before starting training in custom training, make sure u "
                "entered custom training without pressing anything"
            ))
            return

        print('Starting!')