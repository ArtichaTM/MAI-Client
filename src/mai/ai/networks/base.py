from typing import Generator, TYPE_CHECKING
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from mai.functions import _init_weights

if TYPE_CHECKING:
    from mai.capnp.data_classes import ModulesOutputMapping


class ModuleBase(ABC):
    """
    Base class for all modules, hidden or not
    """
    __slots__ = (
        '_enabled', '_loaded',
        '_training', 'power',
        'models_path',
    )
    _enabled: bool
    _loaded: bool
    _training: bool
    power: float

    output_types: tuple[str, ...] = ()
    input_types: tuple[str, ...] = ()

    def __init__(self, models_path: Path | None) -> None:
        assert models_path is None or isinstance(models_path, Path)
        self.models_path = models_path
        self._enabled: bool = False
        self._training = False
        self._loaded = False
        self.power = 0

    def __repr__(self) -> str:
        return (
            f"<Algo {self.name} "
            f"loaded={self.loaded}, enabled={self.enabled}, "
            f"power={self.power: > 1.2f}>"
        )

    @property
    def loaded(self) -> bool:
        """
        This property shows if model loaded
        model can't be enabled without loading first
        """
        return self._loaded

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
        self._training = value

    @property
    def name(self) -> str:
        """
        Display name of model
        """
        return self.get_name()

    @property
    def file_name(self) -> str:
        raise NotImplementedError()

    @classmethod
    def get_name(cls) -> str:
        """
        Display name of model
        """
        return cls.__module__.split('.')[-1]

    def enabled_parameters(self) -> Generator[torch.nn.Parameter, None, None]:
        yield from ()

    def state_dict(self):
        return dict()

    def set_device(self, device: torch.device) -> None:
        pass

    def load(self) -> None:
        assert not self._loaded
        self._loaded = True

    def unload(self, save: bool = True) -> None:
        assert isinstance(save, bool)
        assert self._loaded
        self._loaded = False

    def requires(self) -> set[str]:
        return set()

    @abstractmethod
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
        raise NotImplementedError()

    def copy[T: ModuleBase](self: T) -> T:
        return type(self)(self.models_path)

    def save(self) -> None:
        pass


class NNModuleBase(ModuleBase):
    """
    Base class for all modules, hidden or not
    """
    __slots__ = (
        '_model', '_model_path',
    )
    _enabled: bool
    _model: torch.nn.Module | None
    _training: bool
    power: float

    output_types: tuple[str, ...] = ()
    input_types: tuple[str, ...] = ()

    def __init__(self, models_path: Path | None) -> None:
        assert models_path is None or isinstance(models_path, Path)
        self._enabled: bool = False
        self._training = False
        self._model: torch.nn.Module | None = None
        if models_path is None:
            self._model_path = None
        else:
            self._model_path = models_path / self.file_name
        self.power = 0

    def __repr__(self) -> str:
        return (
            f"<NNM {self.name} from {self._model_path}, "
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

    @classmethod
    def get_name(cls) -> str:
        """
        Display name of model
        """
        return cls.__module__.split('.')[-1]

    def _create_empty(self):
        self._model = self._create()
        with torch.no_grad():
            self._model.apply(_init_weights)

    def _load(self) -> None:
        if self._model_path is None:
            self._create_empty()
        elif self._model_path:
            try:
                self._model = torch.load(self._model_path, weights_only=False)
            except FileNotFoundError:
                print(f"No model at {self._model_path} exists")
            else:
                if self._model is None:
                    print(f"Model {self._model_path} points to incorrect model")
        if self._model is None:
            self._create_empty()
        assert self._model is not None
        # self._model.to(self._mc._device)
        for param in self._model.parameters():
            param.requires_grad = self.training

    def load(self) -> None:
        """ Loads model into CPU/GPU memory """
        assert self._model is None
        assert not self.loaded
        self._load()

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
        if save and self._model_path is not None:
            torch.save(self._model, self._model_path)
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
        if self._model_path is None:
            raise RuntimeError("Can't save model without model path")
        torch.save(self._model, self._model_path)
