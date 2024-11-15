import PySimpleGUI as sg

__all__ = ('MainInterface',)

class MainInterface:
    __slots__ = (
        '_window',
    )
    _window: sg.Window

    def __init__(self):
        self._build_window()

    def _build_window(self) -> None:
        self._window = sg.Window('MAI', [[sg.TabGroup([[
            sg.Tab('Overview', [
                [sg.Text('Overview')]
            ]),
            sg.Tab('Stats', [
                [sg.Text('Stats')]
            ])
        ]], expand_x=True, expand_y=True)]], margins=(0, 0))

    def run(self):
        while True:
            event, values = self._window.read()
            values: dict

            if event == sg.WIN_CLOSED:
                break
