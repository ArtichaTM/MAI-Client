import capnp

from mai.settings import Settings

module = capnp.load(str(
    Settings.path_to_capnp_schemes / 'data.capnp'
))
MAIVector = module.MAIVector
MAIVectorBuilder = MAIVector
MAIVectorReader = MAIVector
MAIRotator = module.MAIRotator
MAIRotatorBuilder = MAIRotator
MAIRotatorReader = MAIRotator
MAIRLObjectState = module.MAIRLObjectState
MAIRLObjectStateBuilder = MAIRLObjectState
MAIRLObjectStateReader = MAIRLObjectState
MAIGameState = module.MAIGameState
MAIGameStateBuilder = MAIGameState
MAIGameStateReader = MAIGameState
MAIControls = module.MAIControls
MAIControlsBuilder = MAIControls
MAIControlsReader = MAIControls
