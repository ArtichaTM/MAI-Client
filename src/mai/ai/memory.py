from collections import deque
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
    __slots__ = ('memory',)

    def __init__(self, capacity: int):
        assert isinstance(capacity, int)
        assert capacity > 1
        self.memory = deque([], maxlen=capacity)

    def push(self, transition: Transition):
        """Save a transition"""
        assert isinstance(transition, Transition)
        if self.memory.maxlen == len(self.memory):
            self.memory.pop()
        self.memory.append(transition)

    def sample(self, batch_size: int):
        assert isinstance(batch_size, int)
        assert batch_size > 0
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
