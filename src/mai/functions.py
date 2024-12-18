from typing import TYPE_CHECKING, Any, Callable, Generator
from threading import current_thread
from functools import partial

import numpy as np
import PySimpleGUI as sg
from live_plotter import FastLivePlotter

if TYPE_CHECKING:
    from mai.ai.rewards.base import NNRewardBase
    from mai.capnp.names import MAIControls, MAIGameState

__all__ = (
    'popup',
    'create_dummy_controls',
)

def _popup(title: str, text: str) -> None:
    assert isinstance(title, str)
    assert isinstance(text, str)
    window = sg.Window(title, [
        [sg.Text(text)],
        [sg.Button('Oк')]
    ])
    window.read()
    window.close()

def popup(title: str, text: str) -> None:
    if current_thread().__class__.__name__ == '_MainThread':
        _popup(title, text)
    else:
        from mai.interface.main_interface import MainInterface
        instance = MainInterface._instance
        assert instance is not None
        instance.call_functions.put_nowait(
            partial(_popup, title, text)
        )

def create_dummy_controls(skip: bool = True) -> 'MAIControls':
    from mai.capnp.names import MAIControls
    controls = MAIControls.new_message()
    controls.skip = skip
    return controls


def rewards_tracker(
    rewards: list['NNRewardBase'],
    on_close: Callable[[], Any] = lambda: ...
) -> Generator[None, 'MAIGameState', None]:
    from mai.capnp.exchanger import Exchanger
    live_plotter = FastLivePlotter(
        n_plots=1,
        titles=["Rewards"],
        xlabels=['i'],
        ylabels=["Reward"],
        xlims=[(0, 200)],
        ylims=[(0, 200)],
        legends=[[i.name for i in rewards]]
    )

    stopping = False

    def _on_close(_):
        nonlocal on_close, stopping
        stopping = True
        on_close()
    live_plotter.fig.canvas.mpl_connect('close_event', _on_close)

    all_rewards: list[list[float]] = []
    while True:
        state = yield
        assert Exchanger._instance is not None
        context = Exchanger._instance.context
        if context is None:
            continue

        all_rewards.append([])
        for cl in rewards:
            all_rewards[-1].append(cl._calculate(state, context))
        if len(all_rewards) > 200:
            all_rewards.pop(0)
        y_data = np.array(all_rewards)
        if stopping:
            break
        live_plotter.plot(
            y_data_list=[y_data],
        )
