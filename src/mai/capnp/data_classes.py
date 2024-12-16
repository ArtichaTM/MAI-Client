from typing import Literal, Type, NamedTuple, TYPE_CHECKING
import enum
from dataclasses import dataclass, field

import numpy as np

from mai.capnp.names import (
    MAIControls,
    MAIGameState,
    MAIVector
)
from mai.settings import Settings

if TYPE_CHECKING:
    import torch
    CAR_OR_BALL = Literal['car'] | Literal['ball']
    V_OR_AV = Literal['v'] | Literal['av']
    MAGNITUDE_OFFSET_TYPING = dict[CAR_OR_BALL, dict[V_OR_AV, float]]


class DodgeVerticalType(enum.IntEnum):
    BACKWARD = -1
    NONE = 0
    FORWARD = 1

class DodgeStrafeType(enum.IntEnum):
    LEFT = -1
    NONE = 0
    RIGHT = 1

class RunType(str, enum.Enum):
    CUSTOM_TRAINING = 'Custom training'
    FREEPLAY = 'Freeplay'
    v11 = '1v1'
    v22 = '2v2'
    v33 = '3v3'
    v44 = '4v4'

class RunParameters(NamedTuple):
    type: RunType
    modules: list[str]
    rewards: dict[str, float]


@dataclass(frozen=False, slots=True, kw_only=False)
class Vector:
    x: float = field(default=0)  # (0, 1)
    y: float = field(default=0)  # (0, 1)
    z: float = field(default=0)  # (0, 1)

    def magnitude(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2)**0.5

    @classmethod
    def from_mai[T:Vector](cls: Type[T], v: MAIVector) -> T:
        return cls(v.x, v.y, v.z)


@dataclass(frozen=True, slots=True, kw_only=True)
class NormalControls:
    throttle: float = field(default=0)
    steer: float = field(default=0)
    pitch: float = field(default=0)
    yaw: float = field(default=0)
    roll: float = field(default=0)
    boost: bool = field(default=False)
    jump: bool = field(default=False)
    handbrake: bool = field(default=False)
    dodgeVertical: DodgeVerticalType = field(default=DodgeVerticalType.NONE)
    dodgeStrafe: DodgeStrafeType = field(default=DodgeStrafeType.NONE)

    def toMAIControls(self) -> MAIControls:
        assert -1 <= self.throttle      <= 1
        assert -1 <= self.steer         <= 1
        assert -1 <= self.pitch         <= 1
        assert -1 <= self.yaw           <= 1
        assert -1 <= self.roll          <= 1
        assert isinstance(self.boost, bool)
        assert isinstance(self.jump, bool)
        assert isinstance(self.handbrake, bool)
        assert isinstance(self.dodgeVertical, DodgeVerticalType)
        assert isinstance(self.dodgeStrafe, DodgeStrafeType)

        controls = MAIControls.new_message()
        print(f"Throttle: {self.throttle: .1f}, steer: {self.steer: .1f}, jump: {self.jump: .1f}")
        if abs(self.throttle) > Settings.control_apply_threshold:
            controls.throttle = self.throttle
        if abs(self.steer) > Settings.control_apply_threshold:
            controls.steer = self.steer
        if abs(self.pitch) > Settings.control_apply_threshold:
            controls.pitch = self.pitch
        if abs(self.yaw) > Settings.control_apply_threshold:
            controls.yaw = self.yaw
        if abs(self.roll) > Settings.control_apply_threshold:
            controls.roll = self.roll
        if self.boost: controls.boost = True
        if self.jump: controls.jump = True
        if self.handbrake: controls.handbrake = True
        if self.dodgeVertical.value != 0: controls.dodgeForward = float(self.dodgeVertical.value)
        if self.dodgeStrafe.value != 0: controls.dodgeStrafe = float(self.dodgeStrafe.value)
        return controls


@dataclass(frozen=True, slots=True, kw_only=True)
class FloatControls:
    throttle: float = field(default=0)
    steer: float = field(default=0)
    pitch: float = field(default=0)
    yaw: float = field(default=0)
    roll: float = field(default=0)
    boost: float = field(default=0)
    jump: float = field(default=0)
    handbrake: float = field(default=0)
    dodgeVertical: float = field(default=0)
    dodgeStrafe: float = field(default=0)

    @staticmethod
    def absMax(value_n: float, value_p: float) -> float:
        """
        absMax(0.5, 0.7) -> 0.7
        absMax(0.7, 0.5) -> -0.7
        absMax(0.7, 0.5) -> -0.7
        absMax(0.5, 0.5) -> 0.5
        """
        assert isinstance(value_n, float)
        assert isinstance(value_p, float)
        assert 0 <= value_n <= 1
        assert 0 <= value_p <= 1
        # return -value_n if abs(value_n) > value_p else value_p
        return value_p-value_n

    def toNormalControls(self) -> NormalControls:
        assert -1 <= self.throttle      <= 1
        assert -1 <= self.steer         <= 1
        assert -1 <= self.pitch         <= 1
        assert -1 <= self.yaw           <= 1
        assert -1 <= self.roll          <= 1
        assert  0 <= self.boost         <= 1
        assert  0 <= self.jump          <= 1
        assert  0 <= self.handbrake     <= 1
        assert -1 <= self.dodgeVertical <= 1
        assert -1 <= self.dodgeStrafe   <= 1

        return NormalControls(
            throttle=self.throttle,
            steer=self.steer,
            pitch=self.pitch,
            yaw=self.yaw,
            roll=self.roll,
            boost=bool(round(self.boost)),
            jump=bool(round(self.jump)),
            handbrake=bool(round(self.handbrake)),
            dodgeVertical=DodgeVerticalType(round(self.dodgeVertical)),
            dodgeStrafe=DodgeStrafeType(round(self.dodgeStrafe))
        )

    @classmethod
    def from_dict_float[T:FloatControls](cls: Type[T], d: dict[str, float]) -> T:
        return cls(
            throttle = d.get('throttle', 0),
            steer = d.get('steer', 0),
            pitch = d.get('pitch', 0),
            yaw = d.get('yaw', 0),
            roll = d.get('roll', 0),
            boost = d.get('boost', 0),
            jump = d.get('jump', 0),
            handbrake = d.get('handbrake', 0),
            dodgeVertical = d.get('dodgeVertical', 0),
            dodgeStrafe = d.get('dodgeStrafe', 0)
        )

    @classmethod
    def from_dict_tensor[T:FloatControls](cls: Type[T], d: dict[str, 'torch.Tensor']) -> T:
        return cls(
            throttle = float(d.get('throttle', 0.0)),
            steer = float(d.get('steer', 0.0)),
            pitch = float(d.get('pitch', 0.0)),
            yaw = float(d.get('yaw', 0.0)),
            roll = float(d.get('roll', 0.0)),
            boost = float(d.get('boost', 0.0)),
            jump = float(d.get('jump', 0.0)),
            handbrake = float(d.get('handbrake', 0.0)),
            dodgeVertical = float(d.get('dodgeVertical', 0.0)),
            dodgeStrafe = float(d.get('dodgeStrafe', 0.0))
        )


@dataclass(frozen=False, slots=True, kw_only=True)
class AdditionalContext:
    """
    Additional context values, that changes based on messages from server
    """
    # Multiplier based on player team
    # If on car spawn y is negative, than we place here -1, else 1
    team_multiplier: Literal[-1, 1] = field(default=1)
    latest_message: MAIGameState.MessageType = field(default='none')
    magnitude_offsets: 'MAGNITUDE_OFFSET_TYPING' = field(
        default_factory=lambda: {'car': {'v': 0, 'av': 0}, 'ball': {'v': 0, 'av': 0}}
    )
