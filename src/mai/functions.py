from typing import TYPE_CHECKING, Any, Callable, Generator
from threading import current_thread
from functools import partial
from multiprocessing import Queue

import numpy as np
import PySimpleGUI as sg

from .plotter import ProcessPlotter

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
) -> Generator[None, 'MAIGameState | None', None]:
    from mai.capnp.exchanger import Exchanger
    process_plotter = ProcessPlotter(tuple((r.name for r in rewards)))
    while True:
        state = yield
        if state is None:
            process_plotter.finish()
            return
        assert Exchanger._instance is not None
        context = Exchanger._instance.context
        if context is None:
            continue

        to_send = []
        for cl in rewards:
            reward = cl._calculate(state, context)
            to_send.append(reward)
        try:
            process_plotter.plot(tuple(to_send))
        except BrokenPipeError:
            on_close()
            return
