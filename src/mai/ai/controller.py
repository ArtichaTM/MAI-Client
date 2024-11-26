from typing import Generator
import numpy as np
from torch import nn, Tensor, tensor
from tensordict.tensordict import TensorDict

from mai.capnp.names import MAIGameState, MAIVector, MAIRotator
from mai.capnp.data_classes import FloatControls
from .networks import build_modules, NNModuleBase


class NNController(nn.Module):
    __slots__ = (
        '_all_modules', '_ordered_modules',
        'current_dict', 'state'
    )
    _all_modules: dict[str, NNModuleBase]
    _ordered_modules: list[NNModuleBase]
    current_dict: TensorDict
    state: MAIGameState
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

    def get_all_modules(self) -> Generator[NNModuleBase, None, None]:
        for module in self._all_modules.values():
            yield module

    def module_enable(self, name: str) -> None:
        assert name not in {i.name for i in self._ordered_modules}
        raise NotImplementedError()

    def module_disable(self, name: str) -> None:
        for index, module in enumerate(self._ordered_modules[1:]):
            if module.name == name:
                break
        else:
            raise KeyError(
                f"There's no module {name}. Possible modules to "
                f"disable: {[i.name for i in self._ordered_modules]}"
            )

        self._ordered_modules.pop(index)
        module_output = set(module.output_types)
        dependencies: list[str] = []
        for next_module in self._ordered_modules[index:]:
            intersection = set(next_module.input_types).intersection(module_output)
            if intersection:
                dependencies.append(next_module.name)
        for dependency_name in dependencies:
            self.module_disable(dependency_name)

    def avg_from_dict(self, name: str) -> float:
        values: list[Tensor] = self.current_dict[name]
        result = sum(values) / len(values)  # type: ignore
        assert isinstance(result, Tensor)
        return float(result)  # type: ignore

    @staticmethod
    def vector_to_tensor(vector: MAIVector) -> Tensor:
        return tensor(list(vector.to_dict().values()))

    @staticmethod
    def rotator_to_tensor(vector: MAIRotator) -> Tensor:
        return tensor(list(vector.to_dict().values()))

    def forward(self) -> FloatControls:
        self.current_dict = TensorDict()  # type: ignore
        for module in self._ordered_modules:
            module.inference(self)

        output = {i: self.avg_from_dict(
            self.current_dict[i]
        ) for i in self.CONTROLS_KEYS}

        return FloatControls.from_dict(output)
