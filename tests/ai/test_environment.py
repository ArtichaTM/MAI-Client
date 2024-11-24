from unittest import TestCase

from torchrl.envs.utils import check_env_specs

from mai.ai.environment import RocketLeagueEnv

class RocketLeagueEnvTests(TestCase):
    def test_load(self):
        env = RocketLeagueEnv()

    def test_check_env_specs(self):
        self.skipTest("Not implemented")
        env = RocketLeagueEnv()
        check_env_specs(env)
