from typing import (
    Literal, Type, NamedTuple, Annotated,
    Mapping, Iterable,
    TYPE_CHECKING
)
import enum
from dataclasses import dataclass, field
from math import tanh, exp

import numpy as np
import torch

from mai.capnp.names import (
    MAIControls,
    MAIGameState,
    MAIVector,
    MAIRotator
)
from mai.settings import Settings

if TYPE_CHECKING:
    from numpy import typing as npt
    import torch
    CAR_OR_BALL = Literal['car'] | Literal['ball']
    V_OR_AV = Literal['v'] | Literal['av']
    MAGNITUDE_OFFSET_TYPING = dict[CAR_OR_BALL, dict[V_OR_AV, float]]
    NumpyVector = Annotated[npt.NDArray[np.float32], Literal[3]]


ARENA_SIZE: 'Vector'
DENORMALIZATION_VECTOR: 'Vector'
STATE_KEYS = (
    'state.car.position.x',
    'state.car.position.y',
    'state.car.position.z',
    'state.car.velocity.x',
    'state.car.velocity.y',
    'state.car.velocity.z',
    'state.car.rotation.pitch',
    'state.car.rotation.roll',
    'state.car.rotation.yaw',
    'state.car.angularVelocity.x',
    'state.car.angularVelocity.y',
    'state.car.angularVelocity.z',
    'state.ball.position.x',
    'state.ball.position.y',
    'state.ball.position.z',
    'state.ball.velocity.x',
    'state.ball.velocity.y',
    'state.ball.velocity.z',
    'state.ball.rotation.pitch',
    'state.ball.rotation.roll',
    'state.ball.rotation.yaw',
    'state.ball.angularVelocity.x',
    'state.ball.angularVelocity.y',
    'state.ball.angularVelocity.z',
    'state.dead',
    'state.boostAmount'
)
CONTROLS_KEYS = (
    'controls.throttle',
    'controls.steer',
    'controls.pitch',
    'controls.yaw',
    'controls.roll',
    'controls.boost',
    'controls.jump',
    'controls.handbrake',
    'controls.dodgeVertical',
    'controls.dodgeStrafe',
)
REWARD_KEY = '_key'

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

class RestartReason(str, enum.Enum):
    BALL_TOUCH = 'Ball touch'
    BALL_EXPLODE = 'Goal explode'
    TIMEOUT = 'Timeout'

class RunParameters(NamedTuple):
    type: RunType
    modules: list[str]
    rewards: Mapping[str, float]
    restart_reasons: set[RestartReason]
    restart_timeout: int
    random_threshold: float
    random_jump: bool

class Vector:
    __slots__ = ('_arr',)

    def __init__(self, arr: 'NumpyVector') -> None:
        self._arr = arr

    def __str__(self) -> str:
        return str(self._arr)

    def __repr__(self) -> str:
        return f"<V {self._arr[0]:.2f} {self._arr[1]:.2f} {self._arr[2]:.2f}>"

    def __sub__(self: 'Vector', other: 'Vector') -> 'Vector':
        return type(self)(self._arr - other._arr)

    def __mul__(self: 'Vector', other: 'Vector') -> 'Vector':
        return type(self)(self._arr * other._arr)

    def __getitem__(self, index: int) -> float:
        assert isinstance(index, int)
        assert 0 <= index <= 2
        return self._arr[index]

    @property
    def x(self) -> float:
        return self._arr[0]

    @property
    def y(self) -> float:
        return self._arr[1]

    @property
    def z(self) -> float:
        return self._arr[2]

    def magnitude(self) -> float:
        return (((
            self._arr*DENORMALIZATION_VECTOR._arr
        )**2).sum())**0.5

    @classmethod
    def from_mai[T:Vector](cls: Type[T], v: MAIVector) -> T:
        return cls(np.array((v.x, v.y, v.z)))

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
    def from_dict_float[T:FloatControls](cls: Type[T], d: Mapping[str, float]) -> T:
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

@dataclass(frozen=False, slots=True, kw_only=True)
class AdditionalContext:
    """
    Additional context values, that changes based on messages from server
    """
    # Multiplier based on player team
    # If on car spawn y is negative, than we place here -1, else 1
    team_multiplier: Literal[-1, 1] = field(default=1)
    latest_message: MAIGameState.MessageType = field(default='none')
    exchanges_since_latest_message: int = field(default=0)
    magnitude_offsets: 'MAGNITUDE_OFFSET_TYPING' = field(
        default_factory=lambda: {'car': {'v': 0, 'av': 0}, 'ball': {'v': 0, 'av': 0}}
    )


class ModulesOutputMapping(dict):
    @classmethod
    def create_random_controls[T: ModulesOutputMapping](
        cls: Type[T],
        /,
        requires_grad: bool = True,
        random_jump: bool = False
    ) -> T:
        assert isinstance(requires_grad, bool)
        assert isinstance(random_jump, bool)
        output = cls({
            key: [torch.rand((), requires_grad=requires_grad)] for key in CONTROLS_KEYS
        })
        if not random_jump:
            output['controls.jump'] = [torch.tensor(0., dtype=torch.get_default_dtype())]
        return output

    @classmethod
    def fromMAIGameState[T: ModulesOutputMapping](
        cls: Type[T],
        s: MAIGameState,
        requires_grad: bool = True
    ) -> T:
        d = dict()
        vtt = cls._vector_to_tensors
        rtt = cls._rotator_to_tensors
        car_position = vtt(s.car.position)
        car_velocity = vtt(s.car.velocity)
        car_rotation = rtt(s.car.rotation)
        car_angularVelocity = vtt(s.car.angularVelocity)
        ball_position = vtt(s.ball.position)
        ball_velocity = vtt(s.ball.velocity)
        ball_rotation = rtt(s.ball.rotation)
        ball_angularVelocity = vtt(s.ball.angularVelocity)
        dead = torch.tensor(
            float(s.dead),
            requires_grad=requires_grad
        )
        boostAmount = torch.tensor(
            float(s.boostAmount),
            requires_grad=requires_grad
        )
        for name, value in [
            ('state.car.position.x', [car_position[0]]),
            ('state.car.position.y', [car_position[1]]),
            ('state.car.position.z', [car_position[2]]),
            ('state.car.velocity.x', [car_velocity[0]]),
            ('state.car.velocity.y', [car_velocity[1]]),
            ('state.car.velocity.z', [car_velocity[2]]),
            ('state.car.rotation.pitch', [car_rotation[0]]),
            ('state.car.rotation.roll', [car_rotation[1]]),
            ('state.car.rotation.yaw', [car_rotation[2]]),
            ('state.car.angularVelocity.x', [car_angularVelocity[0]]),
            ('state.car.angularVelocity.y', [car_angularVelocity[1]]),
            ('state.car.angularVelocity.z', [car_angularVelocity[2]]),
            ('state.ball.position.x', [ball_position[0]]),
            ('state.ball.position.y', [ball_position[1]]),
            ('state.ball.position.z', [ball_position[2]]),
            ('state.ball.velocity.x', [ball_velocity[0]]),
            ('state.ball.velocity.y', [ball_velocity[1]]),
            ('state.ball.velocity.z', [ball_velocity[2]]),
            ('state.ball.rotation.pitch', [ball_rotation[0]]),
            ('state.ball.rotation.roll', [ball_rotation[1]]),
            ('state.ball.rotation.yaw', [ball_rotation[2]]),
            ('state.ball.angularVelocity.x', [ball_angularVelocity[0]]),
            ('state.ball.angularVelocity.y', [ball_angularVelocity[1]]),
            ('state.ball.angularVelocity.z', [ball_angularVelocity[2]]),
            ('state.dead', [dead]),
            ('state.boostAmount', [boostAmount]),
            ('MAIGameState', s)
        ]:
            d[name] = value
        assert set(d.keys()) == set(STATE_KEYS),\
            f"{set(d.keys())}, {set(STATE_KEYS)}"
        return cls(d)

    @classmethod
    def fromTensor[T: ModulesOutputMapping](cls: Type[T], tensor: 'torch.Tensor') -> T:
        assert tensor.shape == (26,), tensor
        return cls([
            ('state.car.position.x', [tensor[0]]),
            ('state.car.position.y', [tensor[1]]),
            ('state.car.position.z', [tensor[2]]),
            ('state.car.velocity.x', [tensor[3]]),
            ('state.car.velocity.y', [tensor[4]]),
            ('state.car.velocity.z', [tensor[5]]),
            ('state.car.rotation.pitch', [tensor[6]]),
            ('state.car.rotation.roll', [tensor[7]]),
            ('state.car.rotation.yaw', [tensor[8]]),
            ('state.car.angularVelocity.x', [tensor[9]]),
            ('state.car.angularVelocity.y', [tensor[10]]),
            ('state.car.angularVelocity.z', [tensor[11]]),
            ('state.ball.position.x', [tensor[12]]),
            ('state.ball.position.y', [tensor[13]]),
            ('state.ball.position.z', [tensor[14]]),
            ('state.ball.velocity.x', [tensor[15]]),
            ('state.ball.velocity.y', [tensor[16]]),
            ('state.ball.velocity.z', [tensor[17]]),
            ('state.ball.rotation.pitch', [tensor[18]]),
            ('state.ball.rotation.roll', [tensor[19]]),
            ('state.ball.rotation.yaw', [tensor[20]]),
            ('state.ball.angularVelocity.x', [tensor[21]]),
            ('state.ball.angularVelocity.y', [tensor[22]]),
            ('state.ball.angularVelocity.z', [tensor[23]]),
            ('state.dead', [tensor[24]]),
            ('state.boostAmount', [tensor[25]]),
        ])

    @staticmethod
    def _vector_to_tensors(
        vector: MAIVector,
        requires_grad: bool = True
    ) -> list['torch.Tensor']:
        return [torch.tensor(
            i, requires_grad=requires_grad
        ) for i in vector.to_dict().values()]

    @staticmethod
    def _rotator_to_tensors(
        rotator: MAIRotator,
        requires_grad: bool = True
    ) -> list['torch.Tensor']:
        return [torch.tensor(
            i, requires_grad=requires_grad
        ) for i in rotator.to_dict().values()]

    def _avg_from_dict(
        self, name: str, requires_grad: bool = True
    ) -> 'torch.Tensor | None':
        values: list[torch.Tensor] = self.get(name, [])
        if not values:
            return None
        assert isinstance(values, list)
        assert all((isinstance(i, torch.Tensor) for i in values)), values
        try:
            tensors = torch.tensor(
                values,
                dtype=torch.float,
                requires_grad=requires_grad
            )
        except RuntimeError:
            print(values)
            raise
        result = tensors.mean()  # type: ignore
        return result

    def has_state(self) -> bool:
        return STATE_KEYS[0] in self

    def has_all_state(self) -> bool:
        return all((key in self for key in STATE_KEYS))

    def has_controls(self) -> bool:
        return CONTROLS_KEYS[0] in self

    def has_all_controls(self) -> bool:
        return all((key in self for key in CONTROLS_KEYS))

    def has_any_controls(self) -> bool:
        return any((key in self for key in CONTROLS_KEYS))

    def has_reward(self) -> bool:
        return REWARD_KEY in self

    @property
    def reward(self) -> float:
        assert self.has_reward()
        return self[REWARD_KEY]

    @reward.setter
    def reward(self, reward: float) -> None:
        assert isinstance(reward, float)
        self[REWARD_KEY] = reward

    @property
    def origin(self) -> 'MAIGameState | None':
        return self.get('MAIGameState', None)

    def is_complete(self) -> bool:
        return self.has_state() and self.has_controls()

    def extract_keys[T: ModulesOutputMapping](
        self: T,
        keys: Iterable[str],
        requires_grad: bool = True
    ) -> T:
        output: Mapping[str, list[torch.Tensor]] = dict()
        for i in keys:
            value = self._avg_from_dict(i, requires_grad=requires_grad)
            if value is not None:
                output[i] = [value]
            else:
                output[i] = [torch.tensor(0.0, requires_grad=requires_grad)]
        return type(self)(output)

    def extract_controls[T: ModulesOutputMapping](
        self: T,
        requires_grad: bool = True
    ) -> T:
        return self.extract_keys(
            keys=CONTROLS_KEYS,
            requires_grad=requires_grad
        )

    def extract_state[T: ModulesOutputMapping](
        self: T,
        requires_grad: bool = True
    ) -> T:
        return self.extract_keys(
            keys=STATE_KEYS,
            requires_grad=requires_grad
        )

    def getNormalizedNegative(self, key: str):
        assert isinstance(key, str)
        v = self.get(key, (0.0,))

        return tanh(float(
            sum(v)/len(v)
        ))

    def getNormalizedPositive(self, key: str):
        assert isinstance(key, str)
        v = self.get(key, (0.0,))
        mean_value = sum(v) / len(v)
        # Clamp the value to avoid overflow
        if mean_value < -709:  # This is a common threshold for float overflow
            return 0.0
        elif mean_value > 709:
            return 1.0
        return 1 / (1 + exp(-mean_value))

    def toFloatControls(self) -> FloatControls:
        return FloatControls(
            throttle = self.getNormalizedNegative('controls.throttle'),
            steer = self.getNormalizedNegative('controls.steer'),
            pitch = self.getNormalizedNegative('controls.pitch'),
            yaw = self.getNormalizedNegative('controls.yaw'),
            roll = self.getNormalizedNegative('controls.roll'),
            boost = self.getNormalizedPositive('controls.boost'),
            jump = self.getNormalizedPositive('controls.jump'),
            handbrake = self.getNormalizedPositive('controls.handbrake'),
            dodgeVertical = self.getNormalizedNegative('controls.dodgeVertical'),
            dodgeStrafe = self.getNormalizedNegative('controls.dodgeStrafe')
        )

    def toTensor(self, requires_grad: bool = True) -> 'torch.Tensor':
        values = [
            sum(i)/len(i) if len(i) > 1 else i[0].item() for i in self.values()
        ]
        return torch.tensor(values, requires_grad=requires_grad)

    def toDiscreteTensor(self, requires_grad: bool = True) -> 'torch.Tensor':
        values = []
        for i in self.values():
            if len(i) > 1:
                v = sum(i)/len(i)
            else:
                v = float(i[0].item())
            values.append(round(v))
        return torch.tensor(values, requires_grad=requires_grad)

    def randomize[T: ModulesOutputMapping](
        self: T, requires_grad: bool = True
    ) -> T:
        for key in self:
            self[key] = [torch.rand((), requires_grad=requires_grad)]
        return self


#
# Constant variables based on data classes
#
ARENA_SIZE = Vector(np.array((4096, 5120+880, 2044)))
DENORMALIZATION_VECTOR = Vector(np.array((
    1,
    ARENA_SIZE.y / ARENA_SIZE.x,
    ARENA_SIZE.z / ARENA_SIZE.z
)))
