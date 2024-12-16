from typing import Any, TYPE_CHECKING, Generator
from abc import ABC, abstractmethod
from pathlib import Path

import torch

if TYPE_CHECKING:
    from ..controller import NNController


class NNModuleBase(ABC):
    """
    Base class for all modules, hidden or not
    """
    __slots__ = (
        '_enabled', '_model', '_training', '_nnc',
        'current_output', 'power'
    )
    _enabled: bool
    _model: torch.nn.Module | None
    _training: bool
    _nnc: 'NNController'
    power: float

    output_types: tuple[str, ...] = ()
    input_types: tuple[str, ...] = ()

    def __init__(self, nnc: 'NNController') -> None:
        assert type(nnc).__qualname__ == 'NNController'
        self._enabled: bool = False
        self._training = False
        self._model: torch.nn.Module | None = None
        self._nnc = nnc
        self.current_output: Any = None
        self.power = 0

    def __repr__(self) -> str:
        return (
            f"<NNM {self.name} from {self.path_to_model}, "
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
            for param in self._model.parameters():
                param.requires_grad = value
        self._training = value

    @property
    def name(self) -> str:
        """
        Display name of model
        """
        return type(self).get_name()


    @classmethod
    def get_name(cls) -> str:
        """
        Display name of model
        """
        return cls.__module__.split('.')[-1]

    @classmethod
    def path_to_model(cls) -> Path:
        return Path(cls.get_name() + '.pt')

    def _init_weights(self, module: torch.nn.Module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.uniform_(module.weight, -1, 1)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.ReLU):
            pass
        elif isinstance(module, torch.nn.Sequential):
            for child in module.children():
                self._init_weights(child)
        else:
            raise RuntimeError(f"Can't initialize layer {module}")

    def enabled_parameters(self) -> Generator[torch.nn.Parameter, None, None]:
        assert self._model is not None
        for parameter in self._model.parameters(recurse=True):
            if parameter.requires_grad:
                yield parameter

    def set_device(self, device: torch.device) -> None:
        assert isinstance(device, torch.device)
        assert self._model is not None
        self._model.to(device)

    def set_training(self, value: bool) -> None:
        assert isinstance(value, bool)
        if not self.loaded: return
        assert self._model is not None
        self._model.training = value
        for parameter in self._model.parameters():
            parameter.requires_grad = value

    def _load(self) -> None:
        path = type(self).path_to_model()
        if path.exists():
            self._model = torch.load(path, weights_only=False)
            if self._model is None:
                raise RuntimeError(
                    f"Model {path} points to incorrect model"
                )
        else:
            self._model = self._create()
            with torch.no_grad():
                self._model.apply(self._init_weights)
            self.save()
        self._model.to(self._nnc._device)
        assert self._model
        for param in self._model.parameters():
            param.requires_grad = self.training

    def load(self) -> None:
        """ Loads model into CPU/GPU memory """
        assert self._model is None
        assert not self.loaded
        self._load()

    def unload(self, save: bool | None) -> None:
        """
        Unloads model from CPU/GPU memory
        :param save: if True, new weights saved into module path
        """
        assert save is None or isinstance(save, bool)
        assert self._model is not None
        assert self.loaded
        if save is None: save = False
        if save:
            path = type(self).path_to_model()
            torch.save(self._model, path)
        self._model = None

    def requires(self) -> set[str]:
        """
        Set of mandatory modules that required to enable this neural network
        """
        return set()

    def inference(self, nnc: 'NNController') -> None:
        """
        Inference some input values
            based on `input_types()`
            and return `output_types()`
        """
        assert self._model is not None
        assert self._enabled
        assert 0 <= self.power <= 1
        # from pprint import pp
        # print('Before: ', end='')
        # pp(nnc.current_dict)
        tensor_dict = nnc.current_dict
        _input = [tensor_dict[i][0] for i in self.input_types]
        print(_input)
        input = torch.tensor(_input)
        print(input)
        output: torch.Tensor = self._model(input) * self.power
        for name, value in zip(self.output_types, output):
            if name not in tensor_dict:
                tensor_dict[name] = []
            tensor_dict[name].append(value)
        # print('After: ', end='')
        # pp(nnc.current_dict)

    @classmethod
    @abstractmethod
    def _create(cls) -> torch.nn.Module:
        raise NotImplementedError()

    def save(self) -> None:
        path = type(self).path_to_model()
        torch.save(self._model, path)
