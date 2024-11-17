import capnp  # type: ignore

from mai.settings import Settings

capnp.remove_import_hook()
module = capnp.load(str(  # type: ignore
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
