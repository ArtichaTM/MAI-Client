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
        # 'controls.steer',
        'controls.pitch',
        'controls.yaw',
        'controls.roll',
        'controls.boost',
        # 'controls.jump',
        # 'controls.handbrake',
        # 'controls.dodgeVertical',
        # 'controls.dodgeStrafe',
    )

    def rotate_car_to_ball(self, car_pos, ball_pos, car_rot):
        """
        Calculates the difference between the desired angle (for XYZ dimensions) and the current car angle
        to rotate the car towards the ball.

        Args:
            car_pos (np.ndarray): Normalized car position (x, y, z).
            ball_pos (np.ndarray): Normalized ball position (x, y, z).
            car_rot (np.ndarray): Car rotation in YZX-ordered Euler angles (y, z, x).

        Returns:
            np.ndarray: Difference between the desired angle and the current car angle (x, y, z).
        """
        # Calculate the desired direction vector from the car to the ball
        desired_dir = ball_pos - car_pos
        desired_dir /= np.linalg.norm(desired_dir)

        # Convert the car's current rotation from Euler angles to a rotation matrix
        car_rot_matrix = np.array([
            [np.cos(car_rot[0]) * np.cos(car_rot[1]),
            np.cos(car_rot[0]) * np.sin(car_rot[1]) * np.sin(car_rot[2]) - np.sin(car_rot[0]) * np.cos(car_rot[2]),
            np.cos(car_rot[0]) * np.sin(car_rot[1]) * np.cos(car_rot[2]) + np.sin(car_rot[0]) * np.sin(car_rot[2])],
            [np.sin(car_rot[0]) * np.cos(car_rot[1]),
            np.sin(car_rot[0]) * np.sin(car_rot[1]) * np.sin(car_rot[2]) + np.cos(car_rot[0]) * np.cos(car_rot[2]),
            np.sin(car_rot[0]) * np.sin(car_rot[1]) * np.cos(car_rot[2]) - np.cos(car_rot[0]) * np.sin(car_rot[2])],
            [-np.sin(car_rot[1]),
            np.cos(car_rot[1]) * np.sin(car_rot[2]),
            np.cos(car_rot[1]) * np.cos(car_rot[2])]
        ])

        # Calculate the difference between the desired direction and the current car direction
        current_dir = car_rot_matrix[:, 0]
        diff = desired_dir - current_dir
        return diff

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
        car_roll = state.car.rotation.roll
        car_yaw = state.car.rotation.yaw
        throttle, pitch, yaw, roll, boost = 1, 0, 0, 0, 0

        # TODO: Add jump if on floor
        # TODO: Add pitch control

        # Yaw
        xy = -car_pos[:-1]
        angle = np.arctan2(xy[1], xy[0])
        target_yaw = angle / np.pi
        diff = (target_yaw-car_yaw + 1) % 2 - 1
        yaw = diff

        # Roll
        roll = -car_roll

        for out_name in self.output_types:
            inner_name = out_name.split('.')[-1]
            v = torch.tensor(locals()[inner_name])
            if out_name in tensor_dict:
                tensor_dict[out_name].append(v)
            else:
                tensor_dict[out_name] = [v]


    def requires(self) -> set[str]:
        return set()
