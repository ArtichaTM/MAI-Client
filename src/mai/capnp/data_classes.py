from dataclasses import dataclass, field
from enum import IntEnum

from mai.capnp.names import MAIControls


class DodgeForwardType(IntEnum):
    BACKWARD = -1
    NONE = 0
    FORWARD = 1

class DodgeStrafeType(IntEnum):
    LEFT = -1
    NONE = 0
    RIGHT = 1


@dataclass(frozen=True, slots=True)
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

    def toMAIControls(self) -> MAIControls:
        controls = MAIControls.new_message()
        controls.throttle = self.throttle
        controls.steer = self.steer
        controls.pitch = self.pitch
        controls.yaw = self.yaw
        controls.roll = self.roll
        controls.boost = self.boost
        controls.jump = self.jump
        controls.handbrake = self.handbrake
        controls.dodgeForward = float(self.dodgeForward.value)
        controls.dodgeStrafe = float(self.dodgeStrafe.value)
        return controls


@dataclass(frozen=True, slots=True)
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
    dodgeStrafe: float = field(default=0)
