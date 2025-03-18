"""This is an automatically generated stub for `data.capnp`."""

import os

import capnp  # type: ignore

capnp.remove_import_hook()  # type: ignore
here = os.path.dirname(os.path.abspath(__file__))
module_file = os.path.abspath(os.path.join(here, "data.capnp"))
module = capnp.load(module_file)  # type: ignore
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
