from logging import warning
from pathlib import Path

from .base import ModuleBase, NNModuleBase


def build_networks() -> dict[str, type[ModuleBase]]:
    output: dict[str, type[ModuleBase]] = dict()
    for file in Path(__file__).parent.iterdir():
        if file.name in {'base.py',} or file.name.startswith('_'):
            continue
        values = globals()
        name = file.name.split('.')[0]
        try:
            exec(f"from .{name} import Module", values)
        except Exception as e:
            warning(
                f"Can't import {name} because:",
                exc_info=e
            )
            continue
        assert 'Module' in values
        module: type[ModuleBase] = values['Module']
        assert issubclass(module, ModuleBase), module
        output[module.get_name()] = module
    return output
