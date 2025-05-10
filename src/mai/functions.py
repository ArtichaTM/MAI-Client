from typing import TYPE_CHECKING, Any, Callable, Generator, Sequence
from threading import current_thread
from functools import partial

import numpy as np
import torch
import PySimpleGUI as sg

from .plotter import (
    ProcessPlotter, PlotterPlot, PlotterBarH
)

if TYPE_CHECKING:
    from mai.ai.rewards.base import NNRewardBase
    from mai.capnp.names import MAIControls, MAIGameState

__all__ = (
    'popup',
    'create_dummy_controls',
    '_init_weights',
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
    process_plotter = ProcessPlotter(
        PlotterPlot,
        plot_names=tuple((
            r.name.replace('_', ' ').title() for r in rewards
        )),
        xlim=(0, 200),
        ylim=(-1.1, 1.1),
        legend=True
    )

    try:
        while True:
            state = yield
            assert Exchanger._instance is not None
            context = Exchanger._instance.context
            if context is None:
                continue
            if state is None:
                to_send = [None] * len(rewards)
            else:
                to_send = []
                for cl in rewards:
                    reward = cl(state, context)
                    to_send.append(reward)

            try:
                process_plotter.plot(tuple(to_send))
            except BrokenPipeError:
                on_close()
                return
    except GeneratorExit:
        process_plotter.finish()

def powers_tracker(
    module_names: Sequence[str],
    on_close: Callable[[], Any] = lambda: ...
) -> Generator[None, Sequence[float], None]:
    from mai.capnp.exchanger import Exchanger
    process_plotter = ProcessPlotter(
        PlotterBarH,
        bar_names=module_names,
    )

    try:
        while True:
            floats = yield
            floats = tuple(floats)
            assert len(floats) == len(module_names)
            assert all(isinstance(i, float) for i in floats)
            try:
                process_plotter.plot(floats)
            except BrokenPipeError:
                on_close()
                return
    except GeneratorExit:
        process_plotter.finish()

def values_tracker(
    names: tuple[str],
    on_close: Callable[[], Any] = lambda: ...,
    **kwargs
):
    """Allows plotting with only names and values"""
    assert isinstance(names, tuple)
    assert all((isinstance(i, str) for i in names))
    assert callable(on_close)
    process_plotter = ProcessPlotter(names, **kwargs)

    try:
        while True:
            values = yield
            assert isinstance(values, tuple)
            assert all((isinstance(i, (float, int)) for i in values))
            try:
                process_plotter.plot(values)
            except BrokenPipeError:
                on_close()
                return
    except GeneratorExit:
        process_plotter.finish()


def _init_weights(module: torch.nn.Module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.uniform_(module.weight, -1, 1)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (
        torch.nn.ReLU,
        torch.nn.LeakyReLU,
        torch.nn.Sigmoid,
        torch.nn.Dropout,
        torch.nn.Tanh,
        torch.nn.LSTM,
        torch.nn.LSTMCell,
        torch.nn.GRU,
        torch.nn.GRUCell,
    )):
        pass
    elif isinstance(module, torch.nn.Sequential):
        for child in module.children():
            _init_weights(child)
    elif hasattr(module, 'init_weights'):
        func = getattr(module, 'init_weights')
        func()
    else:
        raise RuntimeError(f"Can't initialize layer {module}")


class PIDController:
    __slots__ = (
        'kp', 'ki', 'kd', 'dt',
        'error_sum', 'last_error', 'minmax',
        '__call__',
    )

    def __init__(
        self,
        kp: int | float,
        ki: int | float,
        kd: int | float,
        dt: int | float,
        minmax: tuple[int|float, int|float] | None = None,
    ):
        assert isinstance(kp, (int, float))
        assert isinstance(ki, (int, float))
        assert isinstance(kd, (int, float))
        assert isinstance(dt, (int, float))
        assert dt > 0
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.error_sum = 0
        self.last_error = 0
        self.minmax = minmax
        if isinstance(minmax, tuple):
            assert isinstance(minmax, tuple)
            assert isinstance(minmax[0], (int, tuple))
            assert isinstance(minmax[1], (int, tuple))
            self.__call__ = self._windup
        else:
            assert minmax is None
            self.__call__ = self._no_windup

    def _windup(self, error: float) -> float:
        assert isinstance(error, float)
        assert isinstance(self.minmax, tuple)

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.error_sum += error * self.dt
        i_term = self.ki * self.error_sum

        # Derivative term
        d_term = self.kd * (error - self.last_error) / self.dt
        self.last_error = error

        # Calculate the control output
        control_output = p_term + i_term + d_term
        control_output = np.clip(control_output, *self.minmax)

        # Anti-windup
        if self.minmax[0] < control_output < self.minmax[1]:
            self.error_sum -= error * self.dt

        return control_output

    def _no_windup(self, error: float) -> float:
        assert isinstance(error, float)

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.error_sum += error * self.dt
        i_term = self.ki * self.error_sum

        # Derivative term
        d_term = self.kd * (error - self.last_error) / self.dt
        self.last_error = error

        # Calculate the control output
        control_output = p_term + i_term + d_term
        return control_output
