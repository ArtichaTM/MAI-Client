import numpy as np
import torch

from mai.capnp.data_classes import (
    ModulesOutputMapping,
    Vector
)

from .base import ModuleBase


class Module(ModuleBase):
    __slots__ = ()
    input_types = (
        # 'state.car.position.x',
        # 'state.car.position.y',
        # 'state.car.position.z',
        # 'state.car.velocity.x',
        # 'state.car.velocity.y',
        # 'state.car.velocity.z',
        # 'state.car.rotation.pitch',
        # 'state.car.rotation.roll',
        # 'state.car.rotation.yaw',
        # 'state.ball.position.x',
        # 'state.ball.position.y',
        # 'state.ball.position.z',
        # 'state.ball.velocity.x',
        # 'state.ball.velocity.y',
        # 'state.ball.velocity.z',
        # 'state.boostAmount',
    )
    output_types = (
        'controls.throttle',
        'controls.steer',
        # 'controls.pitch',
        'controls.yaw',
        # 'controls.roll',
        'controls.boost',
        # 'controls.jump',
        # 'controls.handbrake',
        # 'controls.dodgeVertical',
        # 'controls.dodgeStrafe',
    )

    def inference(
        self,
        tensor_dict: ModulesOutputMapping,
        requires_grad: bool = False
    ) -> None:
        assert isinstance(tensor_dict, ModulesOutputMapping)
        assert isinstance(requires_grad, bool)
        state = tensor_dict.origin
        assert state is not None
        ball_vector = np.array([
            state.ball.position.x,
            state.ball.position.y,
            state.ball.position.z
        ])
        car_vector = np.array([
            state.car.position.x,
            state.car.position.y,
            state.car.position.z
        ])
        car_rotation = (
            state.car.rotation.pitch,
            state.car.rotation.yaw,
            state.car.rotation.roll
        )
        direction_vector = ball_vector - car_vector

        # Calculate the car's forward vector
        forward_vector = np.array([np.cos(np.pi * car_rotation[1]) * np.cos(np.pi * car_rotation[0]),
                                np.sin(np.pi * car_rotation[0]),
                                np.sin(np.pi * car_rotation[1]) * np.cos(np.pi * car_rotation[0])])

        # Calculate the car's right vector
        right_vector = np.cross(forward_vector, np.array([0, 1, 0]))
        right_vector = right_vector / np.max(np.abs(right_vector))
        throttle = np.dot(direction_vector, forward_vector)
        steer = np.dot(direction_vector, right_vector)
        yaw = np.arctan2(direction_vector[1], direction_vector[0]) / np.pi
        throttle, steer, yaw = map(lambda x: round(x, 4), (
            throttle, steer, yaw
        ))

        throttle = torch.tensor(throttle, requires_grad=requires_grad)
        if 'controls.throttle' in tensor_dict:
            tensor_dict['controls.throttle'].append(throttle)
        else:
            tensor_dict['controls.throttle'] = [throttle]
        steer = torch.tensor(steer, requires_grad=requires_grad)
        if 'controls.steer' in tensor_dict:
            tensor_dict['controls.steer'].append(steer)
        else:
            tensor_dict['controls.steer'] = [steer]
        yaw = torch.tensor(yaw, requires_grad=requires_grad)
        if 'controls.yaw' in tensor_dict:
            tensor_dict['controls.yaw'].append(yaw)
        else:
            tensor_dict['controls.yaw'] = [yaw]

    def requires(self) -> set[str]:
        return set()
