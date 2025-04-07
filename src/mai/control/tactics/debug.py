from .bases import BaseTactic
from mai.capnp.data_classes import (
    MAIGameState,
    AdditionalContext,
    ModulesOutputMapping,
    NormalControls,
)
from mai.functions import create_dummy_controls
from mai.ai.networks.reachBallAir import Module as reachBallAir


class DebugTactic(BaseTactic):
    __slots__ = ('cl',)

    def __init__(self) -> None:
        super().__init__()

    def prepare(self) -> None:
        super().prepare()
        self.cl = reachBallAir(None)

    def react_gen(self):
        state: MAIGameState
        context: AdditionalContext | None
        state, context = yield create_dummy_controls()
        self.cl.load()
        while True:
            mapping = ModulesOutputMapping.fromMAIGameState(state)
            self.cl.inference(mapping, requires_grad=False)
            # controls = create_dummy_controls()
            controls = mapping.toFloatControls().toNormalControls().toMAIControls()
            state, context = yield controls
            assert context is None or isinstance(context, AdditionalContext)
