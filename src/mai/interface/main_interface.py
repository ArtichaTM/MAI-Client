from typing import Optional

import PySimpleGUI as sg

from mai.capnp.exchanger import Exchanger
from mai.capnp.names import MAIControls, MAIGameState

__all__ = ('MainInterface',)


class Constants:
    SLEEP_TIME = '-SLEEP-TIME-'
    TPS_COUNTER = '-TPS-COUNTER-'


class MainInterface:
    __slots__ = (
        '_window',
    )
    _window: sg.Window

    def __init__(self):
        self._build_window()

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
                                default_value=2,
                                resolution=0.01,
                                orientation='horizontal',
                                enable_events=True,
                                k=Constants.SLEEP_TIME
                            )
                        ],
                    ]),
                sg.StatusBar(text="TPS: ", k=Constants.TPS_COUNTER)
            ]
        ], expand_x=True, expand_y=True)]], margins=(0, 0))

    def exchange(self, state: MAIGameState) -> Optional[MAIControls]:
        assert Exchanger._instance is not None
        tps_counter: sg.StatusBar = self._window[Constants.TPS_COUNTER]
        tps_counter.update(f"TPS: {' ':>2}, calls: {Exchanger._instance._exchanges_done}")
        return None

    def run(self):
        exchanger = Exchanger._instance
        Exchanger.register_for_exchange(self.exchange)
        assert exchanger is not None
        while True:
            event, values = self._window.read()
            values: dict

            if event == sg.WIN_CLOSED:
                break

            if event == Constants.SLEEP_TIME:
                exchanger.sleep_time = values[Constants.SLEEP_TIME]

