from typing import Generator, Mapping, TYPE_CHECKING
from pathlib import Path

import torch

from mai.capnp.data_classes import ModulesOutputMapping
from .networks import build_networks, NNModuleBase

if TYPE_CHECKING:
    from mai.capnp.names import MAIGameState


class ModulesController:
    __slots__ = (
        # 
        '_all_modules', '_ordered_modules',

        # Torch mandatory variables
        '_training', '_device',

        # public:
        'state', 'models_folder',
    )
    _all_modules: Mapping[str, NNModuleBase]
    _ordered_modules: list[NNModuleBase]
    _device: torch.device
    models_folder: Path | None

    def __init__(
        self,
        _device: torch.device | None = None,
        models_folder: Path | None = None
    ) -> None:
        super().__init__()

        assert models_folder is None or isinstance(models_folder, Path)
        self.models_folder = models_folder

        if _device is None:
            _device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        self._device = _device

        self._all_modules = {k: m(self.models_folder) for k, m in build_networks().items()}
        self._ordered_modules = []
        self._training = False

    def __call__[T](
        self, value: T
    ) -> T:
        if isinstance(value, ModulesOutputMapping):
            for module in self._ordered_modules:
                module.inference(value)
            return value
        else:
            assert isinstance(value, torch.Tensor)
            assert value.shape != (), value
            assert len(value.shape) == 1
            assert (value.shape[0] % 26) == 0, value.shape
            if value.shape == (26,):
                return self.tensor_inference(value)
            elif (value.shape[0] % 26) == 0:
                v = []
                for i in value.view(-1, 26):
                    v.append(self.tensor_inference(i))
                return torch.cat(v).reshape(-1, 10)  # type: ignore
            else:
                raise RuntimeError()

    def tensor_inference(self, value: torch.Tensor) -> torch.Tensor:
        assert value.shape != (), value
        assert value.shape == (26,), value.shape

        mapping = ModulesOutputMapping.fromTensor(value)
        for module in self._ordered_modules:
            module.inference(mapping)
        output = mapping.extract_controls().toTensor()
        return output

    def save(self) -> None:
        for module in self.get_all_modules():
            module.save()

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, value: bool) -> None:
        assert isinstance(value, bool)
        if self._training == value: return
        for model in self.get_all_modules():
            model.training = value
        self._training = value

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

    def enabled_parameters(self) -> Generator[torch.nn.Parameter, None, None]:
        for module in self.get_all_modules():
            for parameter in module.enabled_parameters():
                yield parameter

    def iter_models(self) -> Generator[torch.nn.Module, None, None]:
        for module in self._ordered_modules:
            assert module._model is not None
            yield module._model

    def get_all_modules(self) -> Generator[NNModuleBase , None, None]:
        for module in self._all_modules.values():
            yield module

    def get_module(self, _module: str) -> NNModuleBase:
        assert isinstance(_module, str)
        module = self._all_modules.get(_module, None)
        if module is None:
            raise KeyError(
                f"Module {_module} is not found. "
                "possible modules: "
                f"{', '.join((i for i in self._all_modules.keys()))}"
            )
        return module

    def module_enable(self, _module: str) -> int:
        """ Enables module, which includes it in `NNController._ordered_modules`
        :param name: Name of the module
        :return: Index of inserted module in `NNController._ordered_modules`
        """
        assert isinstance(_module, str)
        module = self.get_module(_module)
        assert isinstance(module, NNModuleBase), module
        assert module not in {i for i in self._ordered_modules}, \
            f"{self._ordered_modules}{module.enabled}"
        if not module.loaded:
            self.module_load(module.name)
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

    def module_disable(self, _module: str) -> None:
        assert isinstance(_module, str)
        module = self.get_module(_module)
        assert isinstance(module, NNModuleBase), module
        assert module in {i for i in self._ordered_modules}
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

    def module_load(self, _module: str) -> None:
        assert isinstance(_module, str)
        module = self.get_module(_module)
        assert module not in {i for i in self._ordered_modules}
        if not module.loaded:
            module.load()

    def module_unload(
        self,
        _module: str,
        save: bool = False
    ) -> None:
        assert isinstance(_module, str)
        module = self.get_module(_module)
        if module.enabled:
            self.module_disable(module.name)
        assert module not in {i for i in self._ordered_modules}
        module.unload(save=save)

    def unload_all_modules(
        self,
        save: bool = True,
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
            self.module_unload(module.name, save=save)

    def copy[T: ModulesController](
        self: T,
        /,
        copy_models_folder: bool = False,
        copy_device: bool = False
    ) -> T:
        models_folder = None
        if copy_models_folder:
            models_folder = self.models_folder
        device = None
        if copy_device:
            device = self.device
        controller = type(self)(device, models_folder)
        del device, models_folder

        assert isinstance(self._ordered_modules, list)
        for module in self._ordered_modules:
            new_module = type(module)(module._model_path)
            new_module.training = module.training
            new_module.enabled = module.enabled
            new_module.power = module.power
            new_module._model = module._create()
            if module._model is not None:
                new_module._model.load_state_dict(module._model.state_dict(), strict=True)
            controller._ordered_modules.append(new_module)
        return controller
