from logging import warning
from pathlib import Path

from .base import NNRewardBase

def build_rewards() -> dict[str, type[NNRewardBase]]:
    output: dict[str, type[NNRewardBase]] = dict()
    for file in Path(__file__).parent.iterdir():
        if file.name in {'base.py',} or file.name.startswith('_'):
            continue
        values = globals()
        name = file.name.split('.')[0]
        try:
            exec(f"from .{name} import NNReward", values)
        except Exception as e:
            warning(
                f"Can't import {name} because:",
                exc_info=e
            )
            continue
        assert 'NNReward' in values
        rewards: type[NNRewardBase] = values['NNReward']
        assert issubclass(rewards, NNRewardBase), rewards
        output[rewards.get_name()] = rewards
    return output
