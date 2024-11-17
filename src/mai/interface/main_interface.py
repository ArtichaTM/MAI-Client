from typing import Optional
from time import perf_counter

import PySimpleGUI as sg

from mai.capnp.exchanger import Exchanger
from mai.capnp.names import MAIControls, MAIGameState

__all__ = ('MainInterface',)


class Constants:
    SLEEP_TIME = '-SLEEP-TIME-'
    EPS_COUNTER = '-EPS-COUNTER-'
    CALLS_COUNTER = '-CALLS-COUNTER-'


class MainInterface:
    __slots__ = (
        '_window', '_latest_exchange'
    )
    _window: sg.Window | None

    def __init__(self):
        self._build_window()
        self._latest_exchange = perf_counter()

    def _build_window(self) -> None:
        self._window = sg.Window('MAI', [
            [
                sg.TabGroup([[
                    sg.Tab('Overview', [
                        [sg.Text('Overview')],
                        [sg.Button('Update')]
                    ]),
                    sg.Tab('Stats', [
                        [sg.Text('Stats')]
                    ]),
                    sg.Tab('Settings', [
                        [
                            sg.T("Sleep time"),
                            sg.Slider(
                                range=(0, 2),
                                default_value=1.5,
                                resolution=0.01,
                                orientation='horizontal',
                                enable_events=True,
                                k=Constants.SLEEP_TIME
                            )
                        ],
                    ])
                ]], expand_x=True, expand_y=True)
            ], [
                sg.StatusBar(text='EPS: 0', k=Constants.EPS_COUNTER),
                sg.StatusBar(text='Calls: 0', k=Constants.CALLS_COUNTER)
            ]
        ], margins=(0, 0))

    def exchange(self, state: MAIGameState) -> Optional[MAIControls]:
        assert Exchanger._instance is not None
        assert self._window is not None
        (
            self
            ._window[Constants.CALLS_COUNTER]
            .update(f"Calls: {Exchanger._instance._exchanges_done}")  # type: ignore
        )
        current_time = perf_counter()
        (
            self
            ._window[Constants.EPS_COUNTER]
            .update(f"EPS: {1/(current_time - self._latest_exchange):.2f}")  # type: ignore
        )
        self._latest_exchange = current_time
        return None

    def run(self):
        exchanger = Exchanger._instance
        Exchanger.register_for_exchange(self.exchange)
        assert exchanger is not None
        while True:
            assert self._window is not None
            event, values = self._window.read()
            values: dict

            match (event):
                case sg.WIN_CLOSED:
                    return
                case Constants.SLEEP_TIME:
                    exchanger.sleep_time = values[Constants.SLEEP_TIME]
