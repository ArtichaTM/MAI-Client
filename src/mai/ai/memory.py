from typing import Generator

from mai.capnp.data_classes import ModulesOutputMapping

import torch

__all__ = ('Transition', 'ReplayMemory')


class Transition:
    __slots__ = (
        'state', '_state_t',
        'action', '_action_t',
        'next_state', '_next_state_t',
        'reward',
    )

    def __init__(
        self,
        state: ModulesOutputMapping,
        action: ModulesOutputMapping,
        next_state: ModulesOutputMapping,
        reward: float
    ) -> None:
        assert isinstance(state, ModulesOutputMapping)
        assert isinstance(action, ModulesOutputMapping)
        assert isinstance(next_state, ModulesOutputMapping)
        assert isinstance(reward, float)
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self._state_t = None
        self._action_t = None
        self._next_state_t = None

    def __repr__(self) -> str:
        return (
            f"> Transition ({self.reward})"
            f"\n\t{[round(float(i), 2) for i in self.state]}"
            f"\n\t{[round(float(i), 2) for i in self.action]}"
        )

    def __iter__(self):
        return iter((self.state, self.action, self.next_state, self.reward))

    @property
    def state_t(self) -> torch.Tensor:
        if self._state_t is None:
            self._state_t = self.state.toTensor()
        return self._state_t

    @property
    def action_t(self) -> torch.Tensor:
        if self._action_t is None:
            self._action_t = self.action.toTensor()
        return self._action_t

    @property
    def next_state_t(self) -> torch.Tensor:
        if self._next_state_t is None:
            self._next_state_t = self.next_state.toTensor()
        return self._next_state_t


class ReplayMemory:
    __slots__ = ('q', '_max_size')
    q: list[Transition]
    _max_size: int

    def __init__(self, max_size: int | None = None):
        assert max_size is None or isinstance(max_size, int)
        self._max_size = max_size if isinstance(max_size, int) else 9999
        self.q = []

    def __len__(self) -> int:
        return len(self.q)

    def add(self, transition: Transition) -> None:
        if len(self) == self._max_size:
            self.q.pop(0)
        self.q.append(transition)

    def calculate_loss(
        self, /,
        previous_actions_factoring_amount: int | None = None,
        requires_grad: bool = True
    ) -> Generator[torch.Tensor, None, None]:
        raise NotImplementedError()

    def avg_reward(self) -> float:
        return self.sum_reward() / len(self)

    def sum_reward(self) -> float:
        return sum([i.reward for i in self.q])

    def clear(self) -> None:
        self.q = []
