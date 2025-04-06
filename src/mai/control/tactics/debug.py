from .bases import BaseTactic
from mai.capnp.data_classes import (
    MAIGameState,
    AdditionalContext,
    ModulesOutputMapping,
    NormalControls,
)
from mai.functions import create_dummy_controls
from mai.ai.networks.reachBallFloor import Module as reachBallFloor


class DebugTactic(BaseTactic):
    __slots__ = ('cl',)

    def __init__(self) -> None:
        super().__init__()

    def prepare(self) -> None:
        super().prepare()
        self.cl = reachBallFloor(None)

    def react_gen(self):
        state: MAIGameState
        context: AdditionalContext | None
        state, context = yield create_dummy_controls()
        while True:
            mapping = ModulesOutputMapping.fromMAIGameState(state)
            self.cl.inference(mapping, requires_grad=False)
            controls = mapping.toFloatControls().toNormalControls().toMAIControls()
            state, context = yield controls
            assert context is None or isinstance(context, AdditionalContext)
