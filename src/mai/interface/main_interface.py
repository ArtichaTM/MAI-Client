from typing import Optional, Callable
from time import perf_counter

import PySimpleGUI as sg

from mai.capnp.exchanger import Exchanger
from mai.capnp.names import MAIControls, MAIGameState
from mai.control import MainController
from mai.capnp.data_classes import NormalControls, DodgeForwardType, DodgeStrafeType
from mai.control.tactics.simple import SequencedCommands

__all__ = ('MainInterface',)


class Constants:
    TABS = ''
    SLEEP_TIME = ''
    EPS_COUNTER = ''
    CALLS_COUNTER = ''
    LOCKED_STATUS = ''
    STATS_CAR_P_X = ''
    STATS_CAR_P_Y = ''
    STATS_CAR_P_Z = ''
    STATS_BALL_P_X = ''
    STATS_BALL_P_Y = ''
    STATS_BALL_P_Z = ''
    STATS_BOOST = ''
    STATS_DEAD = ''
    DEBUG_CLEAR = ''
    DEBUG_PITCH = ''
    DEBUG_YAW = ''
    DEBUG_ROLL = ''
    DEBUG_JUMP_D_L = ''
    DEBUG_JUMP_D_F = ''
    DEBUG_JUMP_D_R = ''
    DEBUG_JUMP_D_B = ''
    DEBUG_BOOST_BUTTON = ''
    DEBUG_BOOST_INPUT = ''
    DEBUG_JUMP_BUTTON = ''
    DEBUG_JUMP_INPUT = ''
    DEBUG_RESET_TRAINING = ''


i = ''
for i in dir(Constants):
    if i.startswith('__'):
        continue
    setattr(Constants, i, f"-{i.replace('_', '-')}-")
del i

class MainInterface:
    __slots__ = (
        '_window', '_latest_exchange', '_exchange_func',
        '_stats_update_enabled', '_controller'
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
                    ]),
                    sg.Tab('Debug', [
                        [
                            sg.Button("ClearAllCommands", k=Constants.DEBUG_CLEAR),
                        ],
                        [
                            sg.Button("Pitch", k=Constants.DEBUG_PITCH),
                            sg.Button("Yaw", k=Constants.DEBUG_YAW),
                            sg.Button("Roll", k=Constants.DEBUG_ROLL),
                        ],
                        [
                            sg.Button("JumpDodgeLeft", k=Constants.DEBUG_JUMP_D_L),
                            sg.Button("JumpDodgeForward", k=Constants.DEBUG_JUMP_D_F),
                            sg.Button("JumpDodgeRight", k=Constants.DEBUG_JUMP_D_R),
                            sg.Button("JumpDodgeBackward", k=Constants.DEBUG_JUMP_D_B),
                        ],
                        [
                            sg.Button("Boost for", k=Constants.DEBUG_BOOST_BUTTON),
                            sg.Input('300', s=(4, 1), k=Constants.DEBUG_BOOST_INPUT),
                            sg.T('ticks'),
                            sg.Button("Jump for", k=Constants.DEBUG_JUMP_BUTTON),
                            sg.Input('300', s=(4, 1), k=Constants.DEBUG_JUMP_INPUT),
                            sg.T('ticks')
                        ],
                        [
                            sg.Button("Reset training", k=Constants.DEBUG_RESET_TRAINING)
                        ]
                    ])
                ]], expand_x=True, expand_y=True, enable_events=True, k=Constants.TABS)
            ], [
                sg.T("EPS:"),
                sg.StatusBar(
                    text='', size=(5, 1),
                    justification='right', k=Constants.EPS_COUNTER
                ),
                sg.T("Calls:"),
                sg.StatusBar(
                    text='', size=(10, 1),
                    justification='right', k=Constants.CALLS_COUNTER
                ),
                sg.T("Controls locked:"),
                sg.StatusBar(
                    text='', size=(5, 1),
                    justification='right', k=Constants.LOCKED_STATUS
                )
            ]
        ], margins=(0, 0))

    def exchange(self, state: MAIGameState) -> MAIControls:
        assert Exchanger._instance is not None
        assert self._window is not None
        controls = self._controller.react(state)
        self._status_bars_update(state, controls)
        if self._stats_update_enabled: self._stats_update(state)
        return controls

    def _update(self, name: str, value) -> None:
        self._window[name].update(str(value))  # type: ignore

    def _status_bars_update(self, state: MAIGameState, controls: MAIControls) -> None:
        assert Exchanger._instance is not None
        self._update(
            Constants.CALLS_COUNTER,
            Exchanger._instance._exchanges_done
        )
        current_time = perf_counter()
        self._update(
            Constants.EPS_COUNTER,
            f"{1/(current_time - self._latest_exchange):.2f}"
        )
        self._update(
            Constants.LOCKED_STATUS,
            not controls.skip
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
        self._controller = MainController()
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
                case Constants.DEBUG_CLEAR:
                    self._controller.clear_all_tactics()
                case Constants.DEBUG_PITCH:
                    self._controller.add_reaction_tactic(SequencedCommands(
                        NormalControls(jump=True),
                        NormalControls(jump=True, pitch=-1.),
                        NormalControls(jump=True, pitch= 0.),
                        NormalControls(jump=True, pitch= 1.),
                        *[NormalControls()] * 5
                    ))
                case Constants.DEBUG_YAW:
                    self._controller.add_reaction_tactic(SequencedCommands(
                        NormalControls(jump=True),
                        NormalControls(jump=True, yaw=-1.),
                        NormalControls(jump=True, yaw= 0.),
                        NormalControls(jump=True, yaw= 1.),
                        *[NormalControls()] * 5
                    ))
                case Constants.DEBUG_ROLL:
                    self._controller.add_reaction_tactic(SequencedCommands(
                        NormalControls(jump=True),
                        NormalControls(jump=True, roll=-1.),
                        NormalControls(jump=True, roll= 0.),
                        NormalControls(jump=True, roll= 1.),
                        *[NormalControls()] * 5
                    ))
                case Constants.DEBUG_JUMP_D_L:
                    self._controller.add_reaction_tactic(SequencedCommands(
                        NormalControls(jump=True),
                        NormalControls(jump=True),
                        NormalControls(jump=False),
                        NormalControls(jump=True,
                                       dodgeStrafe=DodgeStrafeType.LEFT),
                        *[NormalControls()] * 6
                    ))
                case Constants.DEBUG_JUMP_D_F:
                    self._controller.add_reaction_tactic(SequencedCommands(
                        NormalControls(jump=True),
                        NormalControls(jump=True),
                        NormalControls(jump=False),
                        NormalControls(jump=True,
                                       dodgeForward=DodgeForwardType.FORWARD),
                        *[NormalControls()] * 6
                    ))
                case Constants.DEBUG_JUMP_D_R:
                    self._controller.add_reaction_tactic(SequencedCommands(
                        NormalControls(jump=True),
                        NormalControls(jump=True),
                        NormalControls(jump=False),
                        NormalControls(jump=True,
                                       dodgeStrafe=DodgeStrafeType.RIGHT),
                        *[NormalControls()] * 6
                    ))
                case Constants.DEBUG_JUMP_D_B:
                    self._controller.add_reaction_tactic(SequencedCommands(
                        NormalControls(jump=True),
                        NormalControls(jump=True),
                        NormalControls(jump=False),
                        NormalControls(jump=True,
                                       dodgeForward=DodgeForwardType.BACKWARD),
                        *[NormalControls()] * 6
                    ))
                case Constants.DEBUG_BOOST_BUTTON:
                    boost_input: sg.Input = self._window[
                        Constants.DEBUG_BOOST_INPUT
                    ]  # type: ignore
                    try:
                        boost_tick_amount = int(boost_input.get().strip())
                    except ValueError:
                        boost_tick_amount = 1
                    boost_input.update(boost_tick_amount)
                    self._controller.add_reaction_tactic(SequencedCommands(
                        *([NormalControls(boost=True)]*boost_tick_amount),
                        NormalControls(jump=False)
                    ))
                case Constants.DEBUG_JUMP_BUTTON:
                    jump_input: sg.Input = self._window[
                        Constants.DEBUG_JUMP_INPUT
                    ]  # type: ignore
                    try:
                        jump_tick_amount = int(jump_input.get().strip())
                    except ValueError:
                        jump_tick_amount = 1
                    jump_input.update(jump_tick_amount)
                    self._controller.add_reaction_tactic(SequencedCommands(
                        *([NormalControls(jump=True)]*jump_tick_amount),
                        NormalControls(jump=False)
                    ))
                case Constants.DEBUG_RESET_TRAINING:
                    self._controller.add_reaction_tactic(SequencedCommands(
                        NormalControls(reset=True)
                    ))
