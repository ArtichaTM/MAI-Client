from pathlib import Path

import torch

from mai.capnp.data_classes import RunParameters
from mai.ai.mai_model import MAINet
from .base import BaseTrainingTactic


class MAITrainingTactic(BaseTrainingTactic):
    __slots__ = (
        'mai_path', '_model',
        '_pt_save_path',
    )

    def __init__(
        self,
        run_parameters: RunParameters,
        temp_folder: Path
    ) -> None:
        assert isinstance(run_parameters, RunParameters)
        assert isinstance(temp_folder, Path)
        assert temp_folder.is_dir()
        super().__init__(run_parameters)
        self.mai_path = temp_folder / 'mai'

    def _create_empty(self) -> None:
        self._model = MAINet(len(self._run_parameters.modules))

    def prepare(self) -> None:
        super().prepare()
        self.mai_path.mkdir(exist_ok=True, parents=False)
        pt_name = '_'.join(self._run_parameters.modules) + '.pt'
        pt_save_path = self.mai_path / pt_name
        self._pt_save_path = pt_save_path
        if pt_save_path.exists():
            self._model = torch.load(pt_save_path, weights_only=False)
            if self._model is None:
                print(f"Model {pt_save_path} points to incorrect model")
                self._create_empty()
        else:
            self._create_empty()
        assert isinstance(self._model, torch.nn.Module)
