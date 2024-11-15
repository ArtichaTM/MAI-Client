import capnp  # type: ignore

from mai.settings import Settings


capnp.remove_import_hook()  # type: ignore
module = capnp.load(str(  # type: ignore
    Settings.path_to_capnp_schemes.joinpath('data.capnp')
))
Vector = module.Vector
VectorBuilder = Vector
VectorReader = Vector
Quaternion = module.Quaternion
QuaternionBuilder = Quaternion
QuaternionReader = Quaternion
RLObjectState = module.RLObjectState
RLObjectStateBuilder = RLObjectState
RLObjectStateReader = RLObjectState
GameState = module.GameState
GameStateBuilder = GameState
GameStateReader = GameState
Controls = module.Controls
ControlsBuilder = Controls
ControlsReader = Controls
