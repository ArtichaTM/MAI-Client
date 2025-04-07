import numpy as np
import torch

from mai.capnp.data_classes import (
    ModulesOutputMapping,
    Vector

)
from mai.functions import PIDController

from .base import ModuleBase


class Module(ModuleBase):
    __slots__ = (
        'roll_pid',
        'yaw_pid',
        'pitch_pid',
    )
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
        # 'controls.pitch',
        'controls.yaw',
        'controls.roll',
        'controls.boost',
        'controls.jump',
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

    def normalize_velocity(
        self,
        current: float,
        target: float,
        velocity: float
    ) -> float:
        return current

    def load(self) -> None:
        self.roll_pid = PIDController(
            kp=1.,
            ki=0.01,
            kd=0.9,
            dt=0.1
        )
        self.yaw_pid = PIDController(
            kp=1.5,
            ki=0,
            kd=1.2,
            dt=0.1
        )
        return super().load()

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
        ball_velocity = np.array([
            state.ball.velocity.x,
            state.ball.velocity.y,
            state.ball.velocity.z,
        ])
        car_pos = np.array([
            state.car.position.x,
            state.car.position.y,
            state.car.position.z
        ])
        ball_pos += ball_velocity*0.1
        direction_vector = ball_pos-car_pos
        direction_vector_magnitude = np.linalg.norm(direction_vector)
        car_roll = state.car.rotation.roll
        car_yaw = state.car.rotation.yaw
        car_pitch = state.car.rotation.pitch
        throttle, yaw, roll, boost, jump = 1, 0, 0, 0, 0

        # Jump
        if abs(car_pitch)+abs(car_roll) < 0.1:
            jump = 1

        # Yaw
        angle = np.arctan2(direction_vector[1], direction_vector[0])
        target_yaw = angle / np.pi
        yaw = (target_yaw-car_yaw + 1) % 2 - 1
        yaw = np.clip(self.yaw_pid(yaw), -1., 1.)

        # Roll
        roll = -car_roll
        roll = np.clip(self.roll_pid(roll), -1., 1.)

        # Boost
        sum_rotation = sum((
            abs(yaw), abs(roll),
        ))
        boost = sum_rotation < 0.5

        for out_name in self.output_types:
            inner_name = out_name.split('.')[-1]
            v = torch.tensor(locals()[inner_name])
            if out_name in tensor_dict:
                tensor_dict[out_name].append(v)
            else:
                tensor_dict[out_name] = [v]


    def requires(self) -> set[str]:
        return set()
