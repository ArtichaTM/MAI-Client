from typing import Generator
from time import perf_counter, sleep

from mai.functions import popup
from mai.windows import WindowController
from mai.settings import WinButtons
from mai.ai.trainer import Trainer
from mai.capnp.data_classes import (
    MAIGameState,
    MAIControls,
    AdditionalContext,
    ModulesOutputMapping,
    NormalControls
)
from mai.functions import rewards_tracker
from .base import ModuleTrainingTactic


class CustomTraining(ModuleTrainingTactic):
    __slots__ = ('rewards_plot', )

    def prepare(self) -> None:
        self.rewards_plot = None
        return super().prepare()

    def react_gen(
        self
    ) -> Generator[MAIControls, tuple[MAIGameState, AdditionalContext], None]:
        c = NormalControls().toMAIControls()
        c.skip = True
        state, context = yield c
        del c

        keys = WindowController._instance
        assert keys is not None
        keys.update_window()

        if context is None:
            print(context)
            popup("Error during context evaluation", (
                "For some reason no context is available. That usually\n"
                "happens when user starting training immediately after\n"
                "entering the match. Reloading in custom training should help"
            ))
            return

        if context.latest_message == 'kickoffTimerEnded':
            keys.press_key(WinButtons.RESTART_TRAINING)
        elif context.latest_message != 'kickoffTimerStarted':
            print(context.latest_message)
            popup("Error during training startup", (
                "Trying to run training without entering custom training. \n"
                "Before starting training in custom training, make sure u \n"
                "entered custom training without pressing anything"
            ))
            return

        state, context = yield NormalControls().toMAIControls()

        module_str = self._run_parameters.modules[0]
        # for module in self._mc.get_all_modules():
        #     if module.enabled:
        #         self._mc.module_disable(module)

        # module = self._mc.get_module(module_str)
        # self._mc.module_enable(module_str)
        # module.training = True

        def on_rewards_plot_closed() -> None:
            nonlocal self
            self.rewards_plot = None

        self.rewards_plot = rewards_tracker(
            self._rewards,
            on_close=on_rewards_plot_closed
        )
        next(self.rewards_plot)

        trainer = Trainer(self._run_parameters)
        yield from self.async_wait(trainer.prepare)

        try:
            with trainer:
                keys.press_key(WinButtons.FORWARD)
                while True:
                    start_time = perf_counter()
                    while not self.is_restarting(
                        state,
                        context,
                        time_since_start=start_time
                    ):
                        if self.rewards_plot is not None:
                            try:
                                self.rewards_plot.send(state)
                            except StopIteration:
                                self.rewards_plot = None
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
                    keys.press_key(WinButtons.RESTART_TRAINING)
                    yield from self.async_wait(trainer.epoch_end)
                    self.env_reset()
                    if self.rewards_plot:
                        try:
                            self.rewards_plot.send(None)
                        except StopIteration:
                            self.rewards_plot = None
                    keys.press_key(WinButtons.FORWARD)
                    sleep(0)
        except GeneratorExit:
            if self.rewards_plot:
                self.rewards_plot.close()
            keys.press_key(WinButtons.RESTART_TRAINING)
