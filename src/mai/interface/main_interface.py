from typing import Optional, Callable
from time import perf_counter

import PySimpleGUI as sg

from mai.capnp.exchanger import Exchanger
from mai.capnp.names import MAIControls, MAIGameState

__all__ = ('MainInterface',)


class Constants:
    TABS = ''
    SLEEP_TIME = ''
    EPS_COUNTER = ''
    CALLS_COUNTER = ''
    STATS_CAR_P_X = ''
    STATS_CAR_P_Y = ''
    STATS_CAR_P_Z = ''
    STATS_BALL_P_X = ''
    STATS_BALL_P_Y = ''
    STATS_BALL_P_Z = ''
    STATS_BOOST = ''
    STATS_DEAD = ''

i = ''
for i in dir(Constants):
    if i.startswith('__'):
        continue
    setattr(Constants, i, f"-{i.replace('_', '-')}-")
del i

class MainInterface:
    __slots__ = (
        '_window', '_latest_exchange', '_exchange_func',
        '_stats_update_enabled',
    )
    _window: sg.Window | None
    _exchange_func: Callable[[MAIGameState], Optional[MAIControls]] | None

    def __init__(self):
        self._build_window()
        self._latest_exchange = perf_counter()
        self._exchange_func = None
        self._stats_update_enabled = False

    def _build_window(self) -> None:
        self._window = sg.Window('MAI', [
            [
                sg.TabGroup([[
                    sg.Tab('Overview', [
                        [sg.Text('Overview')],
                        [sg.Button('Update')]
                    ]),
                    sg.Tab('Stats', (
                        [
                            [
                                sg.Text('Car'),
                                sg.StatusBar('', k=Constants.STATS_CAR_P_X),
                                sg.StatusBar('', k=Constants.STATS_CAR_P_Y),
                                sg.StatusBar('', k=Constants.STATS_CAR_P_Z)
                            ], [
                                sg.Text('Ball'),
                                sg.StatusBar('', k=Constants.STATS_BALL_P_X),
                                sg.StatusBar('', k=Constants.STATS_BALL_P_Y),
                                sg.StatusBar('', k=Constants.STATS_BALL_P_Z)
                            ], [
                                sg.Text('Boost'),
                                sg.StatusBar('', k=Constants.STATS_BOOST),
                                sg.Text('Dead?'),
                                sg.StatusBar('', k=Constants.STATS_DEAD)
                            ]
                        ]
                    )),
                    sg.Tab('Settings', [
                        [
                            sg.T("Sleep time"),
                            sg.Slider(
                                range=(0, 0.3),
                                default_value=0.3,
                                resolution=0.005,
                                orientation='horizontal',
                                enable_events=True,
                                k=Constants.SLEEP_TIME
                            )
                        ],
                    ])
                ]], expand_x=True, expand_y=True, enable_events=True, k=Constants.TABS)
            ], [
                sg.StatusBar(text='EPS: 0', k=Constants.EPS_COUNTER),
                sg.StatusBar(text='Calls: 0', k=Constants.CALLS_COUNTER)
            ]
        ], margins=(0, 0))

    def exchange(self, state: MAIGameState) -> Optional[MAIControls]:
        assert Exchanger._instance is not None
        assert self._window is not None
        self._status_bars_update()
        if self._stats_update_enabled: self._stats_update(state)
        return None

    def _update(self, name: str, value) -> None:
        self._window[name].update(str(value))  # type: ignore

    def _status_bars_update(self) -> None:
        assert Exchanger._instance is not None
        self._update(
            Constants.CALLS_COUNTER,
            f"Calls: {Exchanger._instance._exchanges_done}"
        )
        current_time = perf_counter()
        self._update(
            Constants.EPS_COUNTER,
            f"EPS: {1/(current_time - self._latest_exchange):.2f}"
        )
        self._latest_exchange = current_time

    def _stats_update(self, state: MAIGameState) -> None:
        self._update(Constants.STATS_CAR_P_X , state.car.position.x)
        self._update(Constants.STATS_CAR_P_Y , state.car.position.y)
        self._update(Constants.STATS_CAR_P_Z , state.car.position.z)
        self._update(Constants.STATS_BALL_P_X, state.ball.position.x)
        self._update(Constants.STATS_BALL_P_Y, state.ball.position.y)
        self._update(Constants.STATS_BALL_P_Z, state.ball.position.z)
        self._update(Constants.STATS_BOOST, state.boostAmount)
        self._update(Constants.STATS_DEAD, state.dead)

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
                case Constants.TABS:
                    new_tab = values[Constants.TABS]
                    if new_tab == 'Stats':
                        assert not self._stats_update_enabled
                        self._stats_update_enabled = True
                    elif self._stats_update_enabled:
                        self._stats_update_enabled = False
