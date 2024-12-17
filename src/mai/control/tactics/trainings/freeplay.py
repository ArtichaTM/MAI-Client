from .base import ModuleTrainingTactic


class FreeplayTraining(ModuleTrainingTactic):
    __slots__ = ()

    def react_gen(self):
        raise NotImplementedError()
