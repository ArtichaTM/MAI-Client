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
        # 'controls.yaw',
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
        ball_pos = np.array([
            state.ball.position.x,
            state.ball.position.y,
            state.ball.position.z
        ])
        car_pos = np.array([
            state.car.position.x,
            state.car.position.y,
            state.car.position.z
        ])
        car_velocity = np.array([
            state.car.velocity.x,
            state.car.velocity.y,
            state.car.velocity.z,
        ])

        throttle, steer, boost = 1, 0, 0
        direction_vector = ball_pos - car_pos
        if abs(car_velocity.max()) < 0.01:
            steer = 1
        else:
            direction_vector = direction_vector/np.linalg.norm(direction_vector)
            car_velocity = car_velocity/np.linalg.norm(car_velocity)

            dot = np.dot(direction_vector, car_velocity)
            angle = np.arccos(dot)
            cross = np.cross(direction_vector, car_velocity)
            sign = np.sign(cross[2])
            if angle > 2:
                steer = 1
            elif sign > 0:
                steer = -1
            else:
                steer = 1
            if abs(angle) < 1:
                boost = 1-angle

        for out_name in self.output_types:
            inner_name = out_name.split('.')[-1]
            v = torch.tensor(locals()[inner_name])
            if out_name in tensor_dict:
                tensor_dict[out_name].append(v)
            else:
                tensor_dict[out_name] = [v]

    def requires(self) -> set[str]:
        return set()
