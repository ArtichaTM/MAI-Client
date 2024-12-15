import torch

class Transition:
    __slots__ = ('input', 'actions', 'reward', 'next_input')

    def __init__(
        self,
        input: torch.Tensor,
        actions_taken: torch.Tensor,
        reward: float
    ) -> None:
        assert isinstance(input, torch.Tensor)
        assert isinstance(actions_taken, torch.Tensor)
        assert isinstance(reward, float)
        self.input = input
        self.reward = reward
        self.actions = actions_taken

    def complete[T:Transition](self: T, next_input: torch.Tensor) -> T:
        assert isinstance(next_input, torch.Tensor)
        self.next_input = next_input
        return self
