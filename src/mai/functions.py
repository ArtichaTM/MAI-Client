import PySimpleGUI as sg

__all__ = ('popup', )

def popup(title: str, text: str) -> None:
    assert isinstance(title, str)
    assert isinstance(text, str)
    window = sg.Window(title, [
        [sg.Text(text)],
        [sg.Button('Oк')]
    ])
    window.read()
    window.close()
