from typing import TYPE_CHECKING

import PySimpleGUI as sg

if TYPE_CHECKING:
    from mai.capnp.names import MAIControls

__all__ = (
    'popup',
    'create_dummy_controls',
)

def popup(title: str, text: str) -> None:
    assert isinstance(title, str)
    assert isinstance(text, str)
    window = sg.Window(title, [
        [sg.Text(text)],
        [sg.Button('Oк')]
    ])
    window.read()
    window.close()


def create_dummy_controls(skip: bool = True) -> 'MAIControls':
    from mai.capnp.names import MAIControls
    controls = MAIControls.new_message()
    controls.skip = skip
    return controls
