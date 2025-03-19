"""This is an automatically generated stub for `data.capnp`."""

from pathlib import Path

import capnp

capnp.remove_import_hook()  # type: ignore
module = capnp.load(str(Path(__file__).parent / '../../../../CapnP/data.capnp'))  # type: ignore
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
