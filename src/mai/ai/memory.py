from typing import Generator
from collections.abc import Iterator
import random
from dataclasses import dataclass

from mai.capnp.data_classes import ModulesOutputMapping

import torch
import numpy as np

__all__ = ('ReplayMemory',)


@dataclass(slots=True, frozen=True)
class CanonicalValues:
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        yield from (
            self.states, self.actions,
            self.next_states, self.rewards
        )


class ReplayMemory[T: ModulesOutputMapping](list):
    __slots__ = ('_max_size',)
    _max_size: int

    def __init__(self, *args, max_size: int | None = None, **kwargs,):
        super().__init__(*args, **kwargs)
        assert max_size is None or isinstance(max_size, int)
        self._max_size = max_size if isinstance(max_size, int) else 9999

    def __iter__(self) -> Iterator[T]:
        return super().__iter__()

    def add(self, mapping: T) -> None:
        if len(self) == self._max_size:
            self.pop(0)
        self.append(mapping)

    def avg_reward(self) -> float:
        return self.sum_reward() / len(self)

    def sum_reward(self) -> float:
        return sum([i.reward for i in self])

    def sample[S: ReplayMemory](self: S, batch_size: int) -> S:
        assert isinstance(batch_size, int)
        return type(self)(random.sample(self, k=batch_size))

    def to_canonical(self, requires_grad: bool = True) -> CanonicalValues:
        assert len(self) > 1
        v = CanonicalValues(
            torch.stack ([m.extract_state().toTensor() for m in self[:-1]]),
            torch.stack ([m.extract_controls().toTensor() for m in self[:-1]]),
            torch.stack ([m.extract_state().toTensor() for m in self[1:]]),
            torch.tensor(
                [m.reward for m in self[:-1]],
                requires_grad=requires_grad,
                dtype=torch.get_default_dtype()
            )
        )
        assert v.states.dtype == torch.get_default_dtype()
        assert v.actions.dtype == torch.get_default_dtype()
        assert v.next_states.dtype == torch.get_default_dtype()
        assert v.rewards.dtype == torch.get_default_dtype()
        assert len(v.states) == len(v.actions) == len(v.next_states) == len(v.rewards)
        return v

    def rewards_differentiate(self) -> None:
        rewards = [self[0].reward]
        for i in range(1, len(self)):
            rewards.append(self[i].reward - self[i-1].reward)
        assert len(rewards) == len(self)
        for new_reward, mapping in zip(rewards, self):
            mapping.reward = new_reward

    def rewards_normalize(self) -> float:
        """Normalizes all values to [0; 1]
        :return: Multiplier (1/maximum_reward)
        """
        rewards = [i.reward for i in self]
        multiplier = 1/max(rewards)
        for mapping in self:
            mapping.reward *= multiplier
        return multiplier

    def rewards_affect_previous(
        self,
        percent: float = 0.1,
        start_multiplier: float = 0.5
    ) -> None:
        effect_range = int(len(self) * percent)
        assert effect_range > 0
        multipliers = list(np.linspace(0.5, 0, effect_range+1)[:-1])
        for index in range(effect_range, len(self)):
            for reverse_index in range(max(0, index-effect_range), index):
                multiplier = multipliers[index-reverse_index-1]
                diff = self[index].reward - self[reverse_index].reward
                self[reverse_index].reward += diff*multiplier


@dataclass(slots=True)
class Transition:
    state: ModulesOutputMapping
    powers: torch.Tensor


class TensorReplayMemory(list[Transition]):
    __slots__ = ('_max_size',)
    _max_size: int

    def __init__(self, *args, max_size: int | None = None, **kwargs,):
        super().__init__(*args, **kwargs)
        assert max_size is None or isinstance(max_size, int)
        self._max_size = max_size if isinstance(max_size, int) else 9999

    def __iter__(self) -> Iterator[Transition]:
        return super().__iter__()

    def add_v(
        self,
        state: ModulesOutputMapping,
        powers: torch.Tensor
    ):
        assert isinstance(state, ModulesOutputMapping)
        assert isinstance(powers, torch.Tensor)
        self.add_t(Transition(
            state=state,
            powers=powers
        ))

    def add_t(self, transition: Transition) -> None:
        assert isinstance(transition, Transition)
        if len(self) == self._max_size:
            self.pop(0)
        self.append(transition)

    def avg_reward(self) -> float:
        return self.sum_reward() / len(self)

    def sum_reward(self) -> float:
        return sum([i.state.reward for i in self])

    def to_canonical(self, requires_grad: bool = True) -> CanonicalValues:
        assert len(self) > 1
        v = CanonicalValues(
            states=torch.stack ([m.state.extract_state(
                requires_grad
            ).toTensor(requires_grad) for m in self[:-1]]),
            actions=torch.stack ([m.state.extract_controls(
                requires_grad
            ).toTensor(requires_grad) for m in self[:-1]]),
            next_states=torch.stack ([m.state.extract_state(
                requires_grad
            ).toTensor(requires_grad) for m in self[1:]]),
            rewards=torch.tensor(
                [m.state.reward for m in self[1:]],
                requires_grad=requires_grad,
                dtype=torch.get_default_dtype()
            )
        )
        assert v.states.dtype == torch.get_default_dtype()
        assert v.actions.dtype == torch.get_default_dtype()
        assert v.next_states.dtype == torch.get_default_dtype()
        assert v.rewards.dtype == torch.get_default_dtype()
        assert len(v.states)==len(v.actions)==len(v.next_states)==len(v.rewards)
        return v

    def rewards_differentiate(self) -> None:
        rewards = [self[0].state.reward]
        for i in range(1, len(self)):
            rewards.append(self[i].state.reward - self[i-1].state.reward)
        assert len(rewards) == len(self)
        for new_reward, mapping in zip(rewards, self):
            mapping.state.reward = new_reward

    def rewards_normalize(self) -> float:
        """Normalizes all values to [0; 1]
        :return: Multiplier (1/maximum_reward)
        """
        rewards = [i.state.reward for i in self]
        multiplier = 1/max(rewards)
        for mapping in self:
            mapping.state.reward *= multiplier
        return multiplier

    def rewards_affect_previous(
        self,
        percent: float = 0.1
    ) -> None:
        effect_range = int(len(self) * percent)
        assert effect_range > 0
        multipliers = list(np.linspace(0.5, 0, effect_range+1)[:-1])
        for index in range(effect_range, len(self)):
            for reverse_index in range(max(0, index-effect_range), index):
                multiplier = multipliers[index-reverse_index-1]
                diff = self[index].state.reward - self[
                    reverse_index].state.reward
                self[reverse_index].state.reward += diff*multiplier
