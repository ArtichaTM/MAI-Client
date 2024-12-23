from typing import Generator
from queue import SimpleQueue
import random

import torch

__all__ = ('Transition', 'ReplayMemory')


class Transition:
    __slots__ = ('state', 'action', 'next_state', 'reward',)

    def __init__(self, state, action, next_state, reward) -> None:
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

        if __debug__:
            if isinstance(self.state, torch.Tensor):
                assert self.state.shape == (26,), f"{self.state.shape}{self.state}"
            if isinstance(self.action, torch.Tensor):
                assert self.action.shape == (10,), f"{self.action.shape}{self.action}"
            if isinstance(self.next_state, torch.Tensor):
                assert self.next_state.shape == (26,), f"{self.next_state.shape}{self.next_state}"
            if isinstance(self.reward, torch.Tensor):
                assert self.reward.shape == (1,), f"{self.reward.shape}{self.reward}"

    def __repr__(self) -> str:
        return (
            f"> Transition ({self.reward})"
            f"\n\t{[round(float(i), 2) for i in self.state]}"
            f"\n\t{[round(float(i), 2) for i in self.action]}"
        )

    def __iter__(self):
        return iter((self.state, self.action, self.next_state, self.reward))



class ReplayMemory:
    __slots__ = ('q', '_max_size')
    q: SimpleQueue[Transition]
    _max_size: int

    def __init__(self, max_size: int | None = None):
        assert max_size is None or isinstance(max_size, int)
        self._max_size = max_size if isinstance(max_size, int) else 0
        self.q = SimpleQueue()

    def add(self, transition: Transition) -> None:
        if self.q.qsize == self._max_size:
            self.q.get()
        self.q.put(transition)

    def calculate_loss(
        self, /,
        previous_actions_factoring_amount: int | None = None,
        requires_grad: bool = True
    ) -> Generator[torch.Tensor, None, None]:
        raise NotImplementedError()

    def clear(self) -> None:
        self.q = SimpleQueue()
