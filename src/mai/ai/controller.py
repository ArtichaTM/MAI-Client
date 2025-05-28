from io import StringIO
from typing import Generator, Mapping, TYPE_CHECKING
from pathlib import Path

import torch

from mai.capnp.data_classes import ModulesOutputMapping
from .networks import build_networks, ModuleBase, NNModuleBase

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
    _all_modules: Mapping[str, ModuleBase]
    _ordered_modules: list[ModuleBase]
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
                assert any((
                    module.enabled and module.power >= 0.1,
                    not module.enabled and module.power <= 0.1,
                )), module
                module.inference(value)
            return value
        if isinstance(value, list):
            for v in value:
                self(v)
            return value
        else:
            raise RuntimeError(f"Can't inference {type(value)}({value})")

    def __repr__(self) -> str:
        io = StringIO()
        io.write('<MC')
        if self.models_folder is not None:
            io.write(f' path={self.models_folder}')
        loaded = sum((1 if m.loaded else 0 for m in self.get_all_modules()))
        enabled = sum((1 if m.loaded else 0 for m in self.get_all_modules()))
        io.write(f" with {enabled}/{loaded}/{len(self._all_modules)}")
        io.write('>')
        return io.getvalue()

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
            if not module.loaded:
                continue
            for parameter in module.enabled_parameters():
                assert parameter.dtype == torch.get_default_dtype()
                yield parameter

    def iter_models(self) -> Generator[torch.nn.Module, None, None]:
        for module in self._ordered_modules:
            if not isinstance(module, NNModuleBase):
                continue
            assert module._model is not None
            yield module._model

    def get_all_modules(self) -> Generator[ModuleBase , None, None]:
        for module in self._all_modules.values():
            yield module

    def get_module(self, _module: str) -> ModuleBase:
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
        assert isinstance(module, ModuleBase), module
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
        assert isinstance(module, ModuleBase), module
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
        module_name: str,
        save: bool = False
    ) -> None:
        assert isinstance(module_name, str)
        module = self.get_module(module_name)
        if module.enabled:
            self.module_disable(module.name)
        assert module not in {i for i in self._ordered_modules}
        module.unload(save=save)

    def module_power(
        self,
        module_name: str,
        power: float
    ) -> None:
        assert isinstance(module_name, str)
        assert isinstance(power, float)
        assert 0 <= power <= 1, power
        module = self.get_module(module_name)
        if power < 0.1:
            if module.enabled:
                self.module_disable(module_name)
            module.power = power
            return
        module.power = power

    def unload_all_modules(
        self,
        save: bool = True,
        _exceptions: set[ModuleBase | str] | None = None
    ) -> None:
        if _exceptions is None:
            exceptions = set()
        else:
            exceptions = set()
            for exc in _exceptions:
                if isinstance(exc, str):
                    exceptions.add(self.get_module(exc))
                else:
                    assert isinstance(exc, ModuleBase)
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
            new_module = type(module)(module.models_path)
            new_module.training = module.training
            new_module.enabled = module.enabled
            new_module.power = module.power
            if isinstance(module, NNModuleBase):
                assert isinstance(new_module, NNModuleBase)
                new_module._model = module._create()
                if module._model is not None:
                    new_module._model.load_state_dict(
                        module._model.state_dict(), strict=True
                    )
            controller._ordered_modules.append(new_module)
        return controller
