from typing import Generator, TYPE_CHECKING
from torch import Tensor, tensor
from tensordict.tensordict import TensorDict

from mai.capnp.data_classes import FloatControls
from .networks import build_modules, NNModuleBase

if TYPE_CHECKING:
    from mai.capnp.names import MAIGameState, MAIVector, MAIRotator


class NNController:
    __slots__ = (
        '_all_modules', '_ordered_modules', '_training',
        'current_dict', 'state'
    )
    _all_modules: dict[str, NNModuleBase]
    _ordered_modules: list[NNModuleBase]
    current_dict: TensorDict
    state: 'MAIGameState'
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

    def __init__(self) -> None:
        super().__init__()
        self._all_modules = {k: m() for k, m in build_modules().items()}
        assert 'state' in self._all_modules, self._all_modules
        self._ordered_modules = [self._all_modules['state']]
        self.training = False

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, value: bool) -> None:
        assert isinstance(value, bool)
        for model in self.get_all_modules():
            model.set_training(value)
        self._training = value

    def _avg_from_dict(self, name: str) -> float:
        values: list[Tensor] = self.current_dict[name]
        result = sum(values) / len(values)  # type: ignore
        assert isinstance(result, Tensor)
        return float(result)  # type: ignore

    def get_all_modules(self) -> Generator[NNModuleBase, None, None]:
        for module in self._all_modules.values():
            yield module

    def get_module(self, _module: NNModuleBase | str) -> NNModuleBase:
        if isinstance(_module, str):
            module = self._all_modules.get(_module, None)
            raise KeyError(f"Module {_module} is not found")
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
        requirements = module.requires()
        requirements_indexes = []
        index = 1
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
        assert module not in {i for i in self._ordered_modules}
        if module.enabled:
            self.module_disable(module.name)
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
        for module in self._ordered_modules[::-1]:
            if module.name in exceptions:
                continue
            self.module_unload(module, save=save)

    @staticmethod
    def vector_to_tensor(vector: 'MAIVector') -> Tensor:
        return tensor(list(vector.to_dict().values()))

    @staticmethod
    def rotator_to_tensor(vector: 'MAIRotator') -> Tensor:
        return tensor(list(vector.to_dict().values()))

    def fill_tensor_dict(self, s: 'MAIGameState') -> None:
        d = self.current_dict
        vtt = self.vector_to_tensor
        rtt = self.rotator_to_tensor
        for name, value in [
            ('state.car.position', [vtt(s.car.position)]),
            ('state.car.velocity', [vtt(s.car.velocity)]),
            ('state.car.rotation', [rtt(s.car.rotation)]),
            ('state.car.angularVelocity', [vtt(s.car.angularVelocity)]),    
            ('state.ball.position', [vtt(s.car.position)]),
            ('state.ball.velocity', [vtt(s.car.velocity)]),
            ('state.ball.rotation', [rtt(s.car.rotation)]),
            ('state.ball.angularVelocity', [vtt(s.car.angularVelocity)]),
            ('state.dead', [vtt(s.car.angularVelocity)]),
            ('state.boostAmount', [vtt(s.car.angularVelocity)]),
        ]:
            d[name] = value

    def exchange(self, state: 'MAIGameState') -> FloatControls:
        self.current_dict = TensorDict()  # type: ignore
        self.fill_tensor_dict(state)

        for module in self._ordered_modules:
            module.inference(self)

        output = {i: self._avg_from_dict(
            self.current_dict[i]
        ) for i in self.CONTROLS_KEYS}

        return FloatControls.from_dict(output)
