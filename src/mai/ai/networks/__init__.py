from logging import warning
from pathlib import Path

from .base import NNModuleBase

def build_networks() -> dict[str, type[NNModuleBase]]:
    output: dict[str, type[NNModuleBase]] = dict()
    for file in Path(__file__).parent.iterdir():
        if file.name in {'base.py',} or file.name.startswith('_'):
            continue
        values = globals()
        name = file.name.split('.')[0]
        try:
            exec(f"from .{name} import NNModule", values)
        except Exception as e:
            warning(
                f"Can't import {name} because:",
                exc_info=e
            )
            continue
        assert 'NNModule' in values
        module: type[NNModuleBase] = values['NNModule']
        assert issubclass(module, NNModuleBase), module
        output[module.get_name()] = module
    return output
