from typing import Any, Optional, Callable, Generator
from time import perf_counter
from queue import Queue, Empty
from functools import partial
import enum

import PySimpleGUI as sg

from mai.capnp.exchanger import Exchanger
from mai.capnp.names import MAIControls, MAIGameState
from mai.control import MainController
from mai.capnp.data_classes import (
    NormalControls,
    DodgeVerticalType,
    DodgeStrafeType,
    AdditionalContext
)
from mai.control.tactics.simple import SequencedCommands
from mai.ai.controller import NNController, NNModuleBase

__all__ = ('MainInterface',)

FLOAT_MAX_SIZE = 6


class Constants(enum.IntEnum):
    TABS = enum.auto()
    SLEEP_TIME = enum.auto()
    EPS_COUNTER = enum.auto()
    CALLS_COUNTER = enum.auto()
    LOCKED_STATUS = enum.auto()
    LATEST_MESSAGE = enum.auto()
    STATS_BOOST = enum.auto()
    STATS_DEAD = enum.auto()
    DEBUG_CLEAR = enum.auto()
    DEBUG_PITCH = enum.auto()
    DEBUG_YAW = enum.auto()
    DEBUG_ROLL = enum.auto()
    DEBUG_JUMP_D_L = enum.auto()
    DEBUG_JUMP_D_F = enum.auto()
    DEBUG_JUMP_D_R = enum.auto()
    DEBUG_JUMP_D_B = enum.auto()
    DEBUG_BOOST_BUTTON = enum.auto()
    DEBUG_BOOST_INPUT = enum.auto()
    DEBUG_JUMP_BUTTON = enum.auto()
    DEBUG_JUMP_INPUT = enum.auto()
    DEBUG_RESET_TRAINING = enum.auto()
    STATS_CAR_P_X = enum.auto()
    STATS_CAR_P_Y = enum.auto()
    STATS_CAR_P_Z = enum.auto()
    STATS_CAR_R_PITCH = enum.auto()
    STATS_CAR_R_YAW = enum.auto()
    STATS_CAR_R_ROLL = enum.auto()
    STATS_BALL_P_X = enum.auto()
    STATS_BALL_P_Y = enum.auto()
    STATS_BALL_P_Z = enum.auto()
    STATS_CAR_A0_X = enum.auto()
    STATS_CAR_A0_Y = enum.auto()
    STATS_CAR_A0_Z = enum.auto()
    STATS_CAR_A1_X = enum.auto()
    STATS_CAR_A1_Y = enum.auto()
    STATS_CAR_A1_Z = enum.auto()
    STATS_CAR_A2_X = enum.auto()
    STATS_CAR_A2_Y = enum.auto()
    STATS_CAR_A2_Z = enum.auto()
    STATS_CAR_E0_X = enum.auto()
    STATS_CAR_E0_Y = enum.auto()
    STATS_CAR_E0_Z = enum.auto()
    STATS_CAR_E1_X = enum.auto()
    STATS_CAR_E1_Y = enum.auto()
    STATS_CAR_E1_Z = enum.auto()
    STATS_CAR_E2_X = enum.auto()
    STATS_CAR_E2_Y = enum.auto()
    STATS_CAR_E2_Z = enum.auto()
    STATS_CAR_E3_X = enum.auto()
    STATS_CAR_E3_Y = enum.auto()
    STATS_CAR_E3_Z = enum.auto()
    MODULES_CHECKBOX = enum.auto()
    MODULES_POWER = enum.auto()
    USE_MATCH_TYPE = enum.auto()
    USE_BUTTON_TRAIN = enum.auto()
    USE_BUTTON_PLAY = enum.auto()
    USE_BUTTON_PAUSE = enum.auto()
    USE_BUTTON_RESUME = enum.auto()
    USE_BUTTON_STOP = enum.auto()

    def module(self, module: 'NNModuleBase') -> str:
        return f"{self}{module.name.replace('_', '-')}-"

    def module_iter(self, modules: list['NNModuleBase']) -> Generator[str, None, None]:
        for module in modules:
            yield self.module(module)


class RunType(str, enum.Enum):
    CUSTOM_TRAINING = 'Custom training'
    v11 = '1v1'
    v22 = '2v2'
    v33 = '3v3'
    v44 = '4v4'


class MainInterface:
    __slots__ = (
        '_window', '_latest_exchange', '_exchange_func',
        '_stats_update_enabled', '_modules_update_enabled',
        '_controller', '_latest_message',
        '_nnc', '_values',
        'call_functions',
    )
    _window: sg.Window | None
    _exchange_func: Callable[[MAIGameState], Optional[MAIControls]] | None
    call_functions: Queue[Callable]
    module_state_to_color: dict[tuple[bool, bool], str] = {
        (False, False): 'red',
        (True, False): 'yellow',
        (True, True): 'green',
    }

    def __init__(self):
        self._latest_exchange = perf_counter()
        self._exchange_func = None
        self._stats_update_enabled = False
        self._modules_update_enabled = False
        self.call_functions = Queue(1)
        self._nnc = NNController()
        self._build_window()

    def __getitem__(self, value: Constants | str) -> sg.Element:
        return self._window[value]  # type: ignore

    def _set_disabled(self, v: bool, *k: Constants | str) -> None:
        assert self._window is not None
        for key in k:
            el = self[key]
            assert isinstance(el, sg.Element)
            el.update(disabled=v)  # type: ignore

    def build_run_params(self) -> dict[str, Any]:
        assert self._values is not None
        match_type = self._values[Constants.USE_MATCH_TYPE]
        assert isinstance(match_type, str)

        self._set_disabled(
            True,
            Constants.USE_BUTTON_PLAY,
            Constants.USE_BUTTON_TRAIN
        )
        module_checkbox_keys = {
            Constants.MODULES_CHECKBOX.module(m
        ): m for m in self._nnc.get_all_modules()}
        self._set_disabled(
            False,
            Constants.USE_BUTTON_PAUSE,
            Constants.USE_BUTTON_STOP,
            *module_checkbox_keys.keys()
        )
        return {
            'type': RunType(match_type),
            'modules': [v.name for k, v in module_checkbox_keys.items(
                ) if self._values[k]]
        }

    def _build_module_row(self, module: 'NNModuleBase') -> list[sg.Element]:
        return [
            sg.Text(module.name[:15].rjust(15,), p=(0, 0)),
            sg.Checkbox(
                '',
                k=Constants.MODULES_CHECKBOX.module(module),
                p=(0, 1)
            ),
            sg.Progress(
                max_value=100,
                s=(10, 10),
                k=Constants.MODULES_POWER.module(module)
            )
        ]

    def exchange(self, state: MAIGameState, context: AdditionalContext) -> MAIControls:
        assert Exchanger._instance is not None
        assert self._window is not None
        controls = self._controller.react(state, context)
        if not self.call_functions.full():
            self.call_functions.put(partial(self._status_bars_update, state, controls))
        return controls

    def _fmt(self, value: float) -> str:
        return format(value, " > 1.3f")

    def _update(self, name: Constants, value: str) -> None:
        assert isinstance(name, Constants), name
        assert isinstance(value, str), value
        self[name].update(value)  # type: ignore

    def _status_bars_update(self, state: MAIGameState, controls: MAIControls) -> None:
        assert Exchanger._instance is not None
        self._update(
            Constants.CALLS_COUNTER,
            str(Exchanger._instance._exchanges_done)
        )
        current_time = perf_counter()
        self._update(
            Constants.EPS_COUNTER,
            f"{1/(current_time - self._latest_exchange):.2f}"
        )
        self._update(Constants.LOCKED_STATUS, str(not controls.skip))
        if state.message != 'none':
            self._update(Constants.LATEST_MESSAGE, state.message)
        self._latest_exchange = current_time
        if self._stats_update_enabled: self._stats_update(state)
        if self._modules_update_enabled: self._modules_update()

    def _stats_update(self, state: MAIGameState) -> None:
        self._update(Constants.STATS_CAR_P_X    , self._fmt(state.car.position.x    ))
        self._update(Constants.STATS_CAR_P_Y    , self._fmt(state.car.position.y    ))
        self._update(Constants.STATS_CAR_P_Z    , self._fmt(state.car.position.z    ))
        self._update(Constants.STATS_CAR_R_PITCH, self._fmt(state.car.rotation.pitch))
        self._update(Constants.STATS_CAR_R_YAW  , self._fmt(state.car.rotation.yaw  ))
        self._update(Constants.STATS_CAR_R_ROLL , self._fmt(state.car.rotation.roll ))
        self._update(Constants.STATS_BALL_P_X   , self._fmt(state.ball.position.x   ))
        self._update(Constants.STATS_BALL_P_Y   , self._fmt(state.ball.position.y   ))
        self._update(Constants.STATS_BALL_P_Z   , self._fmt(state.ball.position.z   ))
        self._update(Constants.STATS_BOOST      ,       str(state.boostAmount       ))
        self._update(Constants.STATS_DEAD       ,       str(state.dead              ))
        for array, letter in zip((state.otherCars.allies, state.otherCars.enemies), ('A', 'E')):
            for i in range(len(array)):
                car = array[i]
                self._update(getattr(Constants, f'STATS_CAR_{letter}{i}_X'), self._fmt(car.position.x))
                self._update(getattr(Constants, f'STATS_CAR_{letter}{i}_Y'), self._fmt(car.position.y))
                self._update(getattr(Constants, f'STATS_CAR_{letter}{i}_Z'), self._fmt(car.position.z))

    def _modules_update(self) -> None:
        for module in self._nnc.get_all_modules():
            progress_bar = self[Constants.MODULES_POWER.module(module)]
            progress_bar.update(round(module.power*100))
            checkbox = self[Constants.MODULES_CHECKBOX.module(module)]
            checkbox_color = type(self).module_state_to_color[(module.loaded, module.enabled)]
            checkbox.update(checkbox_color=checkbox_color)

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
                                sg.StatusBar('', k=Constants.STATS_CAR_P_X, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_P_Y, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_P_Z, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_R_PITCH, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_R_YAW, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_R_ROLL, size=(FLOAT_MAX_SIZE,1))
                            ], [
                                sg.Text('Ball'),
                                sg.StatusBar('', k=Constants.STATS_BALL_P_X, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_BALL_P_Y, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_BALL_P_Z, size=(FLOAT_MAX_SIZE,1))
                            ], [
                                sg.Text('Boost'),
                                sg.StatusBar('', k=Constants.STATS_BOOST, size=(6, 1)),
                                sg.Text('Dead?'),
                                sg.StatusBar('', k=Constants.STATS_DEAD, size=(5, 1))
                            ], [
                                sg.Text('Ally1'),
                                sg.StatusBar('', k=Constants.STATS_CAR_A0_X, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_A0_Y, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_A0_Z, size=(FLOAT_MAX_SIZE,1))
                            ], [
                                sg.Text('Ally2'),
                                sg.StatusBar('', k=Constants.STATS_CAR_A1_X, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_A1_Y, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_A1_Z, size=(FLOAT_MAX_SIZE,1))
                            ], [
                                sg.Text('Ally3'),
                                sg.StatusBar('', k=Constants.STATS_CAR_A2_X, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_A2_Y, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_A2_Z, size=(FLOAT_MAX_SIZE,1))
                            ], [
                                sg.Text('Enemy1'),
                                sg.StatusBar('', k=Constants.STATS_CAR_E0_X, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_E0_Y, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_E0_Z, size=(FLOAT_MAX_SIZE,1))
                            ], [
                                sg.Text('Enemy2'),
                                sg.StatusBar('', k=Constants.STATS_CAR_E1_X, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_E1_Y, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_E1_Z, size=(FLOAT_MAX_SIZE,1))
                            ], [
                                sg.Text('Enemy3'),
                                sg.StatusBar('', k=Constants.STATS_CAR_E2_X, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_E2_Y, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_E2_Z, size=(FLOAT_MAX_SIZE,1))
                            ], [
                                sg.Text('Enemy4'),
                                sg.StatusBar('', k=Constants.STATS_CAR_E3_X, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_E3_Y, size=(FLOAT_MAX_SIZE,1)),
                                sg.StatusBar('', k=Constants.STATS_CAR_E3_Z, size=(FLOAT_MAX_SIZE,1))
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
                        ], [
                            sg.Button("Pitch", k=Constants.DEBUG_PITCH),
                            sg.Button("Yaw", k=Constants.DEBUG_YAW),
                            sg.Button("Roll", k=Constants.DEBUG_ROLL),
                        ], [
                            sg.Button("JumpDodgeLeft", k=Constants.DEBUG_JUMP_D_L),
                            sg.Button("JumpDodgeForward", k=Constants.DEBUG_JUMP_D_F),
                            sg.Button("JumpDodgeRight", k=Constants.DEBUG_JUMP_D_R),
                            sg.Button("JumpDodgeBackward", k=Constants.DEBUG_JUMP_D_B),
                        ], [
                            sg.Button("Boost for", k=Constants.DEBUG_BOOST_BUTTON),
                            sg.Input('300', s=(4, 1), k=Constants.DEBUG_BOOST_INPUT),
                            sg.T('ticks'),
                            sg.Button("Jump for", k=Constants.DEBUG_JUMP_BUTTON),
                            sg.Input('300', s=(4, 1), k=Constants.DEBUG_JUMP_INPUT),
                            sg.T('ticks')
                        ], [
                            sg.Button("Reset training", k=Constants.DEBUG_RESET_TRAINING)
                        ]
                    ]),
                    sg.Tab('Modules', [
                        self._build_module_row(nn) for nn in self._nnc.get_all_modules()
                    ]),
                    sg.Tab('Use', [
                        [
                            sg.Text("Match type:"),
                            sg.OptionMenu(
                                [i.value for i in RunType],
                                k=Constants.USE_MATCH_TYPE,
                                auto_size_text=False,
                                s=(20, 1),
                                default_value=RunType.CUSTOM_TRAINING.value
                            ),
                        ], [
                            sg.Button("Train", k=Constants.USE_BUTTON_TRAIN),
                            sg.Button("Play", k=Constants.USE_BUTTON_PLAY),
                            sg.Button('Pause', k=Constants.USE_BUTTON_PAUSE, disabled=True),
                            sg.Button('Resume', k=Constants.USE_BUTTON_RESUME, disabled=True),
                            sg.Button('Stop', k=Constants.USE_BUTTON_STOP, disabled=True)
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
                ),
                sg.T("Latest message:"),
                sg.StatusBar(
                    text='none', size=(12, 1),
                    justification='right', k=Constants.LATEST_MESSAGE
                )
            ]
        ], margins=(0, 0))

    def run(self) -> int:
        self._controller = MainController()
        exchanger = Exchanger._instance
        Exchanger.register_for_exchange(self.exchange)
        assert exchanger is not None

        while not exchanger.stop:
            event = 'NONE'
            self._values = None
            while event == 'NONE':
                assert self._window is not None
                event, self._values = self._window.read(timeout=0, timeout_key='NONE')
                alive = exchanger.isAlive()
                if not alive:
                    self._window.close()
                    return 1
                if self.call_functions.not_empty:
                    try:
                        self.call_functions.get(timeout=0)()
                    except Empty:
                        pass
                self._values: dict | None
            if event == sg.WIN_CLOSED:
                return 0
            if self._values is None:
                continue
            assert self._window is not None

            match (event):
                case Constants.SLEEP_TIME:
                    exchanger.sleep_time = self._values[Constants.SLEEP_TIME]
                case Constants.TABS:
                    new_tab = self._values[Constants.TABS]
                    if new_tab == 'Stats':
                        assert not self._stats_update_enabled
                        self._stats_update_enabled = True
                    elif self._stats_update_enabled:
                        self._stats_update_enabled = False
                    if new_tab == 'Modules':
                        assert not self._modules_update_enabled
                        self._modules_update_enabled = True
                    elif self._modules_update_enabled:
                        self._modules_update_enabled = False
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
                                       dodgeVertical=DodgeVerticalType.FORWARD),
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
                                       dodgeVertical=DodgeVerticalType.BACKWARD),
                        *[NormalControls()] * 6
                    ))
                case Constants.DEBUG_BOOST_BUTTON:
                    boost_input = self[
                        Constants.DEBUG_BOOST_INPUT
                    ]
                    assert isinstance(boost_input, sg.Input)
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
                    jump_input: sg.Input = self[
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
                case Constants.USE_BUTTON_TRAIN:
                    self._controller.train(self.build_run_params())
                case Constants.USE_BUTTON_PLAY:
                    self._controller.play(self.build_run_params())
                case Constants.USE_BUTTON_PAUSE:
                    result = self._controller.pause()
                    if result:
                        self._set_disabled(True, Constants.USE_BUTTON_PAUSE)
                        self._set_disabled(False, Constants.USE_BUTTON_RESUME)
                case Constants.USE_BUTTON_RESUME:
                    self._controller.resume()
                    self._set_disabled(True, Constants.USE_BUTTON_RESUME)
                    self._set_disabled(False, Constants.USE_BUTTON_PAUSE)
                case Constants.USE_BUTTON_STOP:
                    self._controller.stop()
                    self._set_disabled(
                        True,
                        Constants.USE_BUTTON_RESUME,
                        Constants.USE_BUTTON_PAUSE,
                        Constants.USE_BUTTON_STOP,
                    )
                    self._set_disabled(
                        False,
                        Constants.USE_BUTTON_PLAY,
                        Constants.USE_BUTTON_TRAIN
                    )
        return 2
