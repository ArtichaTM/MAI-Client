from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from mai.capnp.names import MAIControls


class DodgeForwardType(IntEnum):
    BACKWARD = -1
    NONE = 0
    FORWARD = 1

class DodgeStrafeType(IntEnum):
    LEFT = -1
    NONE = 0
    RIGHT = 1


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
    dodgeForward: DodgeForwardType = field(default=DodgeForwardType.NONE)
    dodgeStrafe: DodgeStrafeType = field(default=DodgeStrafeType.NONE)
    reset: bool = field(default=False)

    def toMAIControls(self) -> MAIControls:
        controls = MAIControls.new_message()
        if self.throttle > 0.01: controls.throttle = self.throttle
        if self.steer > 0.01: controls.steer = self.steer
        if self.pitch > 0.01: controls.pitch = self.pitch
        if self.yaw > 0.01: controls.yaw = self.yaw
        if self.roll > 0.01: controls.roll = self.roll
        if self.boost: controls.boost = True
        if self.jump: controls.jump = True
        if self.handbrake: controls.handbrake = True
        if self.dodgeForward.value != 0: controls.dodgeForward = float(self.dodgeForward.value)
        if self.dodgeStrafe.value != 0: controls.dodgeStrafe = float(self.dodgeStrafe.value)
        if self.reset: controls.reset = True
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
    dodgeForward: float = field(default=0)
    dodgeBackward: float = field(default=0)
    dodgeLeft: float = field(default=0)
    dodgeRight: float = field(default=0)

    def toNormalControls(self) -> NormalControls:
        assert 0 <= self.throttle       <= 1
        assert 0 <= self.steer          <= 1
        assert 0 <= self.pitch          <= 1
        assert 0 <= self.yaw            <= 1
        assert 0 <= self.roll           <= 1
        assert 0 <= self.boost          <= 1
        assert 0 <= self.jump           <= 1
        assert 0 <= self.handbrake      <= 1
        assert 0 <= self.dodgeForward   <= 1
        assert 0 <= self.dodgeBackward  <= 1
        assert 0 <= self.dodgeLeft      <= 1
        assert 0 <= self.dodgeRight     <= 1

        dF = np.array([self.dodgeForward, self.dodgeBackward], dtype=np.float32)
        dS = np.array([self.dodgeForward, self.dodgeBackward], dtype=np.float32)
        dF = dF[np.argmax(np.abs(dF))]
        dS = dS[np.argmax(np.abs(dS))]
        return NormalControls(
            throttle=(self.throttle+1)/2,
            steer=(self.steer+1)/2,
            pitch=(self.pitch+1)/2,
            yaw=(self.yaw+1)/2,
            roll=(self.roll+1)/2,
            boost=bool(round(self.boost)),
            jump=bool(round(self.jump)),
            handbrake=bool(round(self.handbrake)),
            dodgeForward=DodgeForwardType(round(dF)),
            dodgeStrafe=DodgeStrafeType(round(dS))
        )
