from math import exp
import random
from typing import Generator, Mapping, TYPE_CHECKING

import torch

from mai.capnp.data_classes import MAIGameState, ModulesOutputMapping
from .memory import ReplayMemory, Transition
from .rewards import NNRewardBase, build_rewards

if TYPE_CHECKING:
    from .controller import ModulesController

class Trainer:
    __slots__ = (
        '_policy_net', '_all_rewards',
        '_loaded', '_replay_memory',
        '_steps_done', '_gen',

        # Hyperparameters
        'batch_size', 'gamma',
        'eps_start', 'eps_end', 'eps_decay',
        'tau', 'lr',

        # Training parameters
        '_optimizer', '_loss_func', '_target_net',
    )
    _policy_net: 'ModulesController'
    _target_net: 'ModulesController'
    _all_rewards: Mapping[str, 'NNRewardBase']
    _batch_size: int
    _optimizer: torch.optim.Optimizer
    _loss_func: torch.nn.modules.loss._Loss

    def __init__(self, mc: 'ModulesController') -> None:
        self._policy_net = mc
        self._all_rewards = {k: m() for k, m in build_rewards().items()}
        self._loaded = False
        self._replay_memory = ReplayMemory(1000)
        self._steps_done: int = 0

    def __enter__[T: Trainer](self: T) -> T:
        self.hyperparameters_init()
        self._optimizer = torch.optim.AdamW(
            list(self._policy_net.enabled_parameters()),
            lr=self.lr,
            amsgrad=True
        )
        self._loss_func = torch.nn.SmoothL1Loss()
        self._target_net = self._policy_net.copy()
        self._target_net.device = self._policy_net.device
        self._loaded = True
        self._gen = self.inference_gen()
        next(self._gen)
        return self

    def __exit__(self, *args) -> None:
        self._loaded = False

    def hyperparameters_init(self) -> None:
        self.batch_size = 10
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.tau = 0.005
        self.lr = 1e-4

    def _select_action(self, state: ModulesOutputMapping) -> ModulesOutputMapping:
        eps_threshold = (
            self.eps_end + (self.eps_start - self.eps_end) *
            exp(-1. * self._steps_done / self.eps_decay)
        )
        self._steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self._policy_net(state)
        else:
            return ModulesOutputMapping.create_random_controls()

    def _optimize_model(self) -> None:
        assert self._loaded
        if len(self._replay_memory) < self.batch_size:
            return
        transitions = self._replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(
            map(lambda s: s is not None, batch.next_state)
        ), device=self._target_net.device, dtype=torch.bool)
        non_final_next_states = torch.cat([
            s for s in batch.next_state if s is not None
        ])

        assert len(set((tuple(i.shape) for i in batch.state))) == 1, \
            f"{set([i.shape for i in batch.state])}"
        assert len(set((tuple(i.shape) for i in batch.action))) == 1, \
            f"{set([i.shape for i in batch.action])}"
        assert len(set((tuple(i.shape) for i in batch.reward))) == 1, \
            f"{set([i.shape for i in batch.reward])}"
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        action_batch = []
        for state in state_batch:
            action_batch.append(self._policy_net(state))
        action_batch = torch.tensor(action_batch)
        state_action_values = state_batch.gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self._target_net.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self._target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss: torch.Tensor = self._loss_func(state_action_values, expected_state_action_values.unsqueeze(1))

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self._policy_net.enabled_parameters(), 100)
        self._optimizer.step()

    def inference(self, state_map: ModulesOutputMapping, reward: float) -> ModulesOutputMapping:
        return self._gen.send((state_map, reward))

    def inference_gen(self) -> Generator[ModulesOutputMapping, tuple[ModulesOutputMapping, float], None]:
        state_map, reward = yield ModulesOutputMapping.create_random_controls()
        state = state_map.toTensor()

        while True:
            action = self._select_action(state_map)
            observation, reward_f = yield action
            next_state = observation.toTensor()

            reward = torch.tensor([reward_f], device=self._policy_net.device)

            self._replay_memory.push(Transition(
                state=state,
                action=action.extract_controls().toTensor(),
                next_state=next_state,
                reward=reward
            ))
            state = next_state

            self._optimize_model()

            for policy, target in zip(self._policy_net.iter_models(), self._target_net.iter_models()):
                policy_dict = policy.state_dict()
                target_dict = target.state_dict()
                for key in policy_dict:
                    target_dict[key] = policy_dict[key]*self.tau + target_dict[key]*(1-self.tau)
                target.load_state_dict(target_dict)


    def epoch_end(self) -> None:
        pass
