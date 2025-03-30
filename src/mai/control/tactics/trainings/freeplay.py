from mai.functions import popup
from mai.ai.trainer import Trainer
from mai.capnp.data_classes import (
    MAIGameState,
    AdditionalContext,
    ModulesOutputMapping
)
from mai.functions import rewards_tracker
from .base import ModuleTrainingTactic

class FreeplayTraining(ModuleTrainingTactic):
    __slots__ = ('rewards_plot', )

    def prepare(self) -> None:
        self.rewards_plot = None
        return super().prepare()

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
            self.finished = True
            return

        def on_rewards_plot_closed() -> None:
            nonlocal rewards_plot
            rewards_plot = None

        self.rewards_plot = rewards_tracker(
            self._rewards,
            on_close=on_rewards_plot_closed
        )
        next(self.rewards_plot)

        state, context = yield

        with Trainer(self._run_parameters) as trainer:
            while True:
                if self.rewards_plot is not None:
                    try:
                        self.rewards_plot.send(state)
                    except StopIteration:
                        rewards_plot = None
                mapping = ModulesOutputMapping.fromMAIGameState(state)
                reward = self.calculate_reward(state, context)
                mapping.reward = reward
                trainer.inference(mapping)
                state, reward = yield (
                    mapping
                    .toFloatControls()
                    .toNormalControls()
                    .toMAIControls()
                )

    def close(self) -> None:
        if self.rewards_plot is not None:
            try:
                self.rewards_plot.send(None)
            except StopIteration:
                pass
        return super().close()
