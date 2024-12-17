from typing import TYPE_CHECKING
from threading import current_thread
from functools import partial

import PySimpleGUI as sg

if TYPE_CHECKING:
    from mai.capnp.names import MAIControls

__all__ = (
    'popup',
    'create_dummy_controls',
)

def _popup(title: str, text: str) -> None:
    assert isinstance(title, str)
    assert isinstance(text, str)
    window = sg.Window(title, [
        [sg.Text(text)],
        [sg.Button('Oк')]
    ])
    window.read()
    window.close()

def popup(title: str, text: str) -> None:
    if current_thread().__class__.__name__ == '_MainThread':
        _popup(title, text)
    else:
        from mai.interface.main_interface import MainInterface
        instance = MainInterface._instance
        assert instance is not None
        instance.call_functions.put_nowait(
            partial(_popup, title, text)
        )

def create_dummy_controls(skip: bool = True) -> 'MAIControls':
    from mai.capnp.names import MAIControls
    controls = MAIControls.new_message()
    controls.skip = skip
    return controls
