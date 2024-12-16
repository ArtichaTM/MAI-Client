from typing import Any, Generator, Callable, TYPE_CHECKING
import torch

from mai.capnp.data_classes import FloatControls, AdditionalContext
from .networks import build_networks, NNModuleBase
from .rewards import build_rewards, NNRewardBase
from .transition import Transition

if TYPE_CHECKING:
    from mai.capnp.names import MAIGameState, MAIVector, MAIRotator


class NNController:
    __slots__ = (
        # 
        '_all_modules', '_ordered_modules',
        '_all_rewards',

        # Torch mandatory variables
        '_replay_buffer', '_training',
        '_device', '_batch_size', '_sub_transition',
        '_reward', '_optimizer', '_reward_decay',

        # public:
        'current_dict', 'state', 'exchange',
    )
    _all_modules: dict[str, NNModuleBase]
    _ordered_modules: list[NNModuleBase]
    _all_rewards: dict[str, NNRewardBase]
    _ordered_rewards: list[NNRewardBase]
    _device: torch.device
    _batch_size: int
    _sub_transition: Transition | None
    _reward: float
    _reward_decay: float
    _optimizer: torch.optim.Optimizer | None
    _replay_buffer: list[Transition]
    current_dict: dict[str, list[torch.Tensor]]
    state: 'MAIGameState'
    exchange: Callable[['MAIGameState', AdditionalContext], FloatControls]
    CONTROLS_KEYS = (
        'controls.throttle',
        'controls.steer',
        'controls.pitch',
        'controls.yaw',
        'controls.roll',
        'controls.boost',
        'controls.jump',
        'controls.handbrake',
        'controls.dodgeVertical',
        'controls.dodgeStrafe',
    )

    def __init__(self, _device: torch.device | None = None) -> None:
        super().__init__()
        from torchrl.data import ReplayBuffer, LazyTensorStorage
        self._all_modules = {k: m(self) for k, m in build_networks().items()}
        self._ordered_modules = []
        self._all_rewards = {k: m() for k, m in build_rewards().items()}
        if _device is None:
            _device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        self._device = _device
        self._batch_size = 32
        self._reward_decay = 1/5
        self._training = False
        self._optimizer = None
        self._replay_buffer = []

        # Loop variables
        self._sub_transition = None
        self._reward = 0.0

    @property
    def batch_size(self) -> int:
        assert isinstance(self._batch_size, int)
        assert self._batch_size >= 8
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        assert isinstance(value, int)
        assert value >= 8
        self._batch_size = value

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, value: bool) -> None:
        assert isinstance(value, bool)
        if self._training == value: return
        for model in self.get_all_modules():
            model.set_training(value)
        self._training = value
        if value:
            self.exchange = self._exchange_train
        else:
            self.exchange = self._exchange_run

    @property
    def current_reward(self) -> float:
        return self._reward

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, value: torch.device) -> None:
        assert isinstance(value, torch.device)
        if self._device == value: return
        self._device = value
        for module in self.get_all_modules():
            module.set_device(value)

    def _avg_from_dict(self, name: str) -> torch.Tensor | None:
        values: list[torch.Tensor] = self.current_dict.get(name, [])
        if not values:
            return None
        tensors = torch.tensor(values)
        result = tensors.mean()  # type: ignore
        assert isinstance(result, torch.Tensor)
        return result

    def prepare_for_training(
        self,
        **kwargs: Any
    ) -> None:
        self.training = True
        self._optimizer = torch.optim.Adam(
            list(self.enabled_parameters()),
            **kwargs
        )

    def enabled_parameters(self) -> Generator[torch.nn.Parameter, None, None]:
        for module in self.get_all_modules():
            for parameter in module.enabled_parameters():
                yield parameter

    def get_all_modules(self) -> Generator[NNModuleBase , None, None]:
        for module in self._all_modules.values():
            yield module

    def get_all_rewards(self) -> Generator[NNRewardBase, None, None]:
        for reward in self._all_rewards.values():
            yield reward

    def get_module(self, _module: NNModuleBase | str) -> NNModuleBase:
        if isinstance(_module, str):
            module = self._all_modules.get(_module, None)
            if module is None:
                raise KeyError(
                    f"Module {_module} is not found. "
                    "possible modules: "
                    f"{', '.join((i for i in self._all_modules.keys()))}"
                )
        else:
            module = _module
        return module

    def module_enable(self, _module: NNModuleBase | str) -> int:
        """ Enables module, which includes it in `NNController._ordered_modules`
        :param name: Name of the module
        :return: Index of inserted module in `NNController._ordered_modules`
        :rtype: int
        """
        module = self.get_module(_module)
        assert module not in {i for i in self._ordered_modules}
        if not module.loaded:
            self.module_load(module)
        requirements = module.requires()
        requirements_indexes = []
        index = 0
        while requirements:
            required_module_name = requirements.pop()
            assert required_module_name in self._all_modules.keys()
            found_module = None
            for module in self._ordered_modules[index:]:
                index += 1
                if module.name == required_module_name:
                    found_module = module
                    break
            if not found_module:
                index = self.module_enable(required_module_name) + 1
            requirements_indexes = [i if i < index else (
                i+1
            ) for i in requirements_indexes]
            requirements_indexes.append(index)
        if requirements_indexes:
            index = max(requirements_indexes) + 1
        self._ordered_modules.insert(index, module)
        module.enabled = True
        return index

    def module_disable(self, _module: NNModuleBase | str) -> None:
        module = self.get_module(_module)
        assert module not in {i for i in self._ordered_modules}
        index = self._ordered_modules.index(module)

        self._ordered_modules.pop(index)
        module_output = set(module.output_types)
        dependencies: list[str] = []
        for next_module in self._ordered_modules[index:]:
            intersection = set(next_module.input_types).intersection(module_output)
            if intersection:
                dependencies.append(next_module.name)
        for dependency_name in dependencies:
            self.module_disable(dependency_name)
        module.enabled = False

    def module_load(self, _module: NNModuleBase | str) -> None:
        module = self.get_module(_module)
        assert module not in {i for i in self._ordered_modules}
        if not module.loaded:
            module.load()

    def module_unload(
        self,
        _module: NNModuleBase | str,
        save: bool | None = None
    ) -> None:
        module = self.get_module(_module)
        if module.enabled:
            self.module_disable(module)
        assert module not in {i for i in self._ordered_modules}
        module.unload(save=save)

    def unload_all_modules(
        self,
        save: bool | None = None,
        _exceptions: set[NNModuleBase | str] | None = None
    ) -> None:
        if _exceptions is None:
            exceptions = set()
        else:
            exceptions = set()
            for exc in _exceptions:
                if isinstance(exc, str):
                    exceptions.add(self.get_module(exc))
                else:
                    assert isinstance(exc, NNModuleBase)
                    exceptions.add(exc)
        for module in self._ordered_modules[::-1].copy():
            if module.name in exceptions:
                continue
            if not module.loaded:
                continue
            self.module_unload(module, save=save)

    @staticmethod
    def vector_to_tensors(vector: 'MAIVector') -> list[torch.Tensor]:
        return [torch.tensor(i) for i in vector.to_dict().values()]

    @staticmethod
    def rotator_to_tensors(rotator: 'MAIRotator') -> list[torch.Tensor]:
        return [torch.tensor(i) for i in rotator.to_dict().values()]

    def fill_tensor_dict(self, s: 'MAIGameState') -> None:
        d = self.current_dict
        vtt = self.vector_to_tensors
        rtt = self.rotator_to_tensors
        car_position = self.vector_to_tensors(s.car.position)
        car_velocity = self.vector_to_tensors(s.car.velocity)
        car_rotation = self.rotator_to_tensors(s.car.rotation)
        car_angularVelocity = self.vector_to_tensors(s.car.angularVelocity)
        ball_position = self.vector_to_tensors(s.ball.position)
        ball_velocity = self.vector_to_tensors(s.ball.velocity)
        ball_rotation = self.rotator_to_tensors(s.ball.rotation)
        ball_angularVelocity = self.vector_to_tensors(s.ball.angularVelocity)
        dead = torch.tensor(s.dead)
        boostAmount = torch.tensor(s.boostAmount)
        for name, value in [
            ('state.car.position.x', [car_position[0]]),
            ('state.car.position.y', [car_position[1]]),
            ('state.car.position.z', [car_position[2]]),
            ('state.car.velocity.x', [car_velocity[0]]),
            ('state.car.velocity.y', [car_velocity[1]]),
            ('state.car.velocity.z', [car_velocity[2]]),
            ('state.car.rotation.pitch', [car_rotation[0]]),
            ('state.car.rotation.roll', [car_rotation[1]]),
            ('state.car.rotation.yaw', [car_rotation[2]]),
            ('state.car.angularVelocity.x', [car_angularVelocity[0]]),
            ('state.car.angularVelocity.y', [car_angularVelocity[1]]),
            ('state.car.angularVelocity.z', [car_angularVelocity[2]]),
            ('state.ball.position.x', [ball_position[0]]),
            ('state.ball.position.y', [ball_position[1]]),
            ('state.ball.position.z', [ball_position[2]]),
            ('state.ball.velocity.x', [ball_velocity[0]]),
            ('state.ball.velocity.y', [ball_velocity[1]]),
            ('state.ball.velocity.z', [ball_velocity[2]]),
            ('state.ball.rotation.pitch', [ball_rotation[0]]),
            ('state.ball.rotation.roll', [ball_rotation[1]]),
            ('state.ball.rotation.yaw', [ball_rotation[2]]),
            ('state.ball.angularVelocity.x', [ball_angularVelocity[0]]),
            ('state.ball.angularVelocity.y', [ball_angularVelocity[1]]),
            ('state.ball.angularVelocity.z', [ball_angularVelocity[2]]),
            ('state.dead', [dead]),
            ('state.boostAmount', [boostAmount]),
        ]:
            d[name] = value

    def calculate_reward(
        self,
        state: 'MAIGameState',
        context: AdditionalContext
    ) -> None:
        current: float = 0.0
        for reward in self._all_rewards.values():
            current += reward(state, context)
        # prev = self._reward
        # self._reward = prev + (current - prev) * self._reward_decay
        self._reward = current

    def _train_epoch(self) -> None:
        pass

    def _exchange_run(self, state: 'MAIGameState', context: AdditionalContext) -> FloatControls:
        self.current_dict = dict()  # type: ignore
        self.fill_tensor_dict(state)

        output: dict[str, torch.Tensor] = dict()
        for i in self.CONTROLS_KEYS:
            value = self._avg_from_dict(i)
            if value is not None:
                output[i] = value
        return FloatControls.from_dict_tensor(output)

    def _exchange_train(self, state: 'MAIGameState', context: AdditionalContext) -> FloatControls:
        self.current_dict = dict()  # type: ignore
        self.fill_tensor_dict(state)

        input_tensor = torch.cat([t[0] for t in self.current_dict.values()])

        for module in self._ordered_modules:
            module.inference(self)

        output: dict[str, torch.Tensor] = dict()
        for i in self.CONTROLS_KEYS:
            value = self._avg_from_dict(i)
            if value is not None:
                output[i] = value

        self.calculate_reward(state, context)

        if self._sub_transition is not None:
            previous_transition = self._sub_transition.complete(input_tensor)
            self._replay_buffer.append(previous_transition)

        self._sub_transition = Transition(
            input=input_tensor,
            reward=self._reward,
            actions_taken=torch.tensor(list(output.values()))
        )
        return FloatControls.from_dict_tensor(output)

    def save(self) -> None:
        for module in self.get_all_modules():
            module.save()
