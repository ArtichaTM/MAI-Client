from unittest import TestCase

from mai.capnp.names import *

class Convert(TestCase):
    def test_controls(self):
        values = {
            'throttle': 0.4,
            'steer': -0.3,
            'pitch': +0.7,
            'yaw': -1.,
            'roll': 1.,
            'boost': True,
            'jump': False
        }
        c: ControlsBuilder = Controls.new_message()
        for key, value in values.items():
            setattr(c, key, value)
        c_bytes = c.to_bytes()

        with Controls.from_bytes(c_bytes) as output:
            for key, value in values.items():
                if isinstance(value, float):
                    self.assertAlmostEqual(getattr(output, key), value)
                elif isinstance(value, bool):
                    self.assertEqual(getattr(output, key), value)
                else:
                    self.fail(f"Unknown type {type(value)} in values")

    def test_game_state(self):
        ball_rotation = -7.5
        ball_position = 5.3
        car_angular_z = -13.65
        state: GameStateBuilder = GameState.new_message(
            car=RLObjectState.new_message(
                position=Vector.new_message(
                    x=0, y=0, z=0
                ),
                speed=Vector.new_message(
                    x=0, y=0, z=0
                ),
                rotation=Vector.new_message(
                    x=0, y=0, z=0
                ),
                angularSpeed=Vector.new_message(
                    x=0, y=0, z=car_angular_z
                )
            ),
            ball=RLObjectState.new_message(
                position=Vector.new_message(
                    x=0, y=ball_position, z=0
                ),
                speed=Vector.new_message(
                    x=0, y=0, z=0
                ),
                rotation=Vector.new_message(
                    x=ball_rotation, y=0, z=0
                ),
                angularSpeed=Vector.new_message(
                    x=0, y=0, z=0
                )
            ),
            boostAmount=55,
            dead=False
        )
        state_bytes = state.to_bytes()

        with GameState.from_bytes(state_bytes) as output:
            self.assertAlmostEqual(state.ball.position.y, ball_position, 6)
            self.assertAlmostEqual(state.boostAmount, output.boostAmount, 6)
            self.assertAlmostEqual(state.ball.rotation.x, ball_rotation, 6)
            self.assertAlmostEqual(state.car.angularSpeed.z, car_angular_z, 6)
