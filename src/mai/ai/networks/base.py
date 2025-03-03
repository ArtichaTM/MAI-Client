from typing import Any, Generator, MutableMapping, TYPE_CHECKING
from abc import ABC, abstractmethod
from pathlib import Path

import torch

if TYPE_CHECKING:
    from ..controller import ModulesController
    from mai.capnp.data_classes import ModulesOutputMapping


class NNModuleBase(ABC):
    """
    Base class for all modules, hidden or not
    """
    __slots__ = (
        '_enabled', '_model', '_training',
        '_mc', 'power'
    )
    _enabled: bool
    _model: torch.nn.Module | None
    _training: bool
    _mc: 'ModulesController'
    power: float

    output_types: tuple[str, ...] = ()
    input_types: tuple[str, ...] = ()

    def __init__(self, mc: 'ModulesController') -> None:
        assert type(mc).__qualname__ == 'ModulesController', mc
        self._enabled: bool = False
        self._training = False
        self._model: torch.nn.Module | None = None
        self._mc = mc
        self.power = 0

    def __repr__(self) -> str:
        return (
            f"<NNM {self.name} from {self.path_to_model()}, "
            f"loaded={self.loaded}, enabled={self.enabled}, "
            f"power={self.power: > 1.2f}>"
        )

    @property
    def loaded(self) -> bool:
        """
        This property shows if model loaded
        model can't be enabled without loading first
        """
        return self._model is not None

    @property
    def enabled(self) -> bool:
        """
        This property shows if model used in generating actions
        """
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._enabled = value

    @property
    def training(self) -> bool:
        """
        If false, weights is frozen
        """
        return self._training

    @training.setter
    def training(self, value: bool) -> None:
        if self._model:
            for param in self._model.parameters(recurse=True):
                param.requires_grad = value
        self._training = value

    @property
    def name(self) -> str:
        """
        Display name of model
        """
        return self.get_name()

    @property
    def file_name(self) -> str:
        return self.get_name() + '.pt'

    def path_to_model(self) -> Path | None:
        if self._mc.models_folder is None:
            return None
        return self._mc.models_folder / self.file_name

    @classmethod
    def get_name(cls) -> str:
        """
        Display name of model
        """
        return cls.__module__.split('.')[-1]

    def _init_weights(self, module: torch.nn.Module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.uniform_(module.weight, -1, 1)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (
            torch.nn.ReLU,
            torch.nn.Sigmoid
        )):
            pass
        elif isinstance(module, torch.nn.Sequential):
            for child in module.children():
                self._init_weights(child)
        else:
            raise RuntimeError(f"Can't initialize layer {module}")

    def _create_empty(self):
        self._model = self._create()
        with torch.no_grad():
            self._model.apply(self._init_weights)

    def _load(self) -> None:
        path = self.path_to_model()
        if path is None:
            self._create_empty()
        elif path.exists():
            self._model = torch.load(path, weights_only=False)
            if self._model is None:
                print(f"Model {path} points to incorrect model")
                self._create_empty()
        else:
            self._create_empty()
        assert self._model is not None
        self._model.to(self._mc._device)
        for param in self._model.parameters():
            param.requires_grad = self.training

    def load(self) -> None:
        """ Loads model into CPU/GPU memory """
        assert self._model is None
        assert not self.loaded
        self._load()

    def copy[T: NNModuleBase](self: T, mc: 'ModulesController | None' = None) -> T:
        if mc is None:
            mc = self._mc
        module = type(self)(mc)
        module.training = self.training
        module.enabled = self.enabled
        module.power = self.power
        module._model = self._create()
        assert self._model is not None
        module._model.load_state_dict(self._model.state_dict(), strict=True)
        return module

    def enabled_parameters(self) -> Generator[torch.nn.Parameter, None, None]:
        assert self._model is not None
        for parameter in self._model.parameters(recurse=True):
            if parameter.requires_grad:
                yield parameter

    def state_dict(self):
        assert self._model is not None
        return self._model.state_dict()

    def set_device(self, device: torch.device) -> None:
        assert isinstance(device, torch.device)
        assert self._model is not None
        self._model.to(device)

    def unload(self, save: bool = True) -> None:
        """
        Unloads model from CPU/GPU memory
        :param save: if True, new weights saved into module path
        """
        assert isinstance(save, bool)
        assert self._model is not None
        assert self.loaded
        path = self.path_to_model()
        if save and path is not None:
            torch.save(self._model, path)
        self._model = None

    def requires(self) -> set[str]:
        """
        Set of mandatory modules that required to enable this neural network
        """
        return set()

    def inference(
        self,
        tensor_dict: 'ModulesOutputMapping',
        requires_grad: bool = False
    ) -> None:
        """
        Inference some input values
            based on `input_types()`
            and return `output_types()`
        """
        assert self._model is not None
        assert self._enabled
        assert 0 <= self.power <= 1
        _input = [tensor_dict[i][0] for i in self.input_types]
        input = torch.tensor(_input, requires_grad=requires_grad)
        output: torch.Tensor = self._model(input) * self.power
        for name, value in zip(self.output_types, output):
            if name not in tensor_dict:
                tensor_dict[name] = []
            tensor_dict[name].append(value)

    @classmethod
    @abstractmethod
    def _create(cls) -> torch.nn.Module:
        raise NotImplementedError()

    def save(self) -> None:
        path = self.path_to_model()
        assert path is not None
        torch.save(self._model, path)
