from typing import Generator, Iterable, NamedTuple
from collections.abc import Iterator
import random

from mai.capnp.data_classes import ModulesOutputMapping

import torch

__all__ = ('Transition', 'ReplayMemory')


class CanonicalValues(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor


class Transition:
    __slots__ = (
        'state', '_state_t',
        'next_state', '_next_state_t',
    )

    def __init__(
        self,
        state: ModulesOutputMapping,
        next_state: ModulesOutputMapping,
    ) -> None:
        assert isinstance(state, ModulesOutputMapping)
        assert isinstance(next_state, ModulesOutputMapping)
        assert state.has_state()
        assert state.has_controls()
        assert state.has_reward()
        assert next_state.has_state()
        assert not next_state.has_controls()
        assert next_state.has_reward()
        self.state = state
        self.next_state = next_state
        self._state_t = None
        self._next_state_t = None

    def __repr__(self) -> str:
        return (
            f"> Transition ()"
            f"\n\t{[round(float(i), 2) for i in self.state]}"
        )

    def __iter__(self):
        return iter((self.state, self.next_state))

    @property
    def state_t(self) -> torch.Tensor:
        if self._state_t is None:
            self._state_t = self.state.toTensor()
            assert self._state_t.dtype == torch.get_default_dtype()
        return self._state_t

    @property
    def next_state_t(self) -> torch.Tensor:
        if self._next_state_t is None:
            self._next_state_t = self.next_state.toTensor()
        return self._next_state_t


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

    def calculate_loss(
        self, /,
        previous_actions_factoring_amount: int | None = None,
        requires_grad: bool = True
    ) -> Generator[torch.Tensor, None, None]:
        raise NotImplementedError()

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
