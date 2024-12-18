from math import exp
import random
from typing import Generator, Mapping, TYPE_CHECKING

import torch

from mai.capnp.data_classes import ModulesOutputMapping
from .memory import ReplayMemory, Transition
from .rewards import NNRewardBase, build_rewards

if TYPE_CHECKING:
    from .controller import ModulesController

class Critic(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(state_size + action_size, 400)
        self.fc2 = torch.nn.Linear(400, 300)
        self.fc3 = torch.nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], 1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Trainer:
    __slots__ = (
        '_actor', 'critic',
        '_all_rewards',
        '_loaded', '_replay_memory',
        '_steps_done', '_gen',

        # Hyperparameters
        'batch_size', 'gamma',
        'eps_start', 'eps_end', 'eps_decay',
        'tau', 'actor_lr', 'critic_lr',

        # Training parameters
        '_actor_optimizer', '_critic_optimizer',
        '_loss_func',
    )
    _actor: 'ModulesController'
    _critic: Critic
    _all_rewards: Mapping[str, 'NNRewardBase']
    _batch_size: int
    _optimizer: torch.optim.Optimizer
    _loss_func: torch.nn.modules.loss._Loss

    def __init__(self, mc: 'ModulesController') -> None:
        self._actor = mc
        self._all_rewards = {k: m() for k, m in build_rewards().items()}
        self._loaded = False
        self._replay_memory = ReplayMemory(1000)
        self._steps_done: int = 0

    def __enter__[T: Trainer](self: T) -> T:
        self.hyperparameters_init()
        self._actor_optimizer = torch.optim.Adam(self._actor.enabled_parameters(), lr=self.actor_lr)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=self.critic_lr)
        # self._optimizer = torch.optim.AdamW(
        #     list(self._actor.enabled_parameters()),
        #     lr=self.lr,
        #     amsgrad=True
        # )
        # self._loss_func = torch.nn.SmoothL1Loss()
        # self._target_net = self._actor.copy()
        # self._target_net.device = self._actor.device
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
        self.actor_lr = 1e-4
        self.critic_lr = 2e-4

    def _select_action(self, state: ModulesOutputMapping) -> ModulesOutputMapping:
        eps_threshold = (
            self.eps_end + (self.eps_start - self.eps_end) *
            exp(-1. * self._steps_done / self.eps_decay)
        )
        self._steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self._actor(state)
        else:
            return ModulesOutputMapping.create_random_controls()

    def _optimize_model(self) -> None:
        assert self._loaded
        if len(self._replay_memory) < self.batch_size:
            return
        transitions = self._replay_memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(
            lambda s: s is not None, batch.next_state
        )), device=self._actor.device, dtype=torch.bool)
        non_final_next_states = torch.cat([
            s for s in batch.next_state if s is not None
        ])
        state_batch = torch.cat(batch.state)
        assert (state_batch.shape[0] % 26) == 0
        action_batch = torch.cat(batch.action)
        assert (action_batch.shape[0] % 10) == 0
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        temp = self._actor(state_batch)
        # if action_batch.dim() == 1:
        #     action_batch = action_batch.unsqueeze(1)
        state_action_values = temp.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self._actor.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self._target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = self._loss_func(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self._actor.enabled_parameters(), 100)
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

            reward = torch.tensor([reward_f], device=self._actor.device)

            self._replay_memory.push(Transition(
                state=state,
                action=action.extract_controls().toTensor(),
                next_state=next_state,
                reward=reward
            ))
            state = next_state

            self._optimize_model()

            for policy, target in zip(self._actor.iter_models(), self._target_net.iter_models()):
                policy_dict = policy.state_dict()
                target_dict = target.state_dict()
                for key in policy_dict:
                    target_dict[key] = policy_dict[key]*self.tau + target_dict[key]*(1-self.tau)
                target.load_state_dict(target_dict)


    def epoch_end(self) -> None:
        pass
