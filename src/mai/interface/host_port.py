import PySimpleGUI as sg

from mai.functions import popup
from mai.settings import Settings

__all__ = ('HostPort',)

class HostPort:
    __slots__ = (
        '_window', 'host', 'port'
    )
    _window: sg.Window

    def __init__(self):
        self.host, self.port = Settings.server_address

    def _build_window(self) -> None:
        self._window = sg.Window('MAI', [
            [sg.Text('Enter Rocket League host:port', pad=(0, 0, 0, 5), font=('Comic Sans', 10))],
            [
                sg.InputText(default_text=Settings.server_address[0], focus=True, size=(25, 10), pad=0, border_width=0),
                sg.Text(':', pad=0),
                sg.InputText(default_text=Settings.server_address[1], size=(5, 10), pad=0, border_width=0)
            ],
            [sg.Submit(pad=(0, 5, 0, 0), font=('Comic Sans', 10))]
        ])

    def run(self) -> bool:
        self._build_window()
        while True:
            event, values = self._window.read()
            values: dict

            if event == sg.WIN_CLOSED:
                self.host, self.port = None, None
                return False

            if event == 'Submit':
                host, port = values.values()
                self.host = host
                try:
                    self.port = int(port)
                except ValueError:
                    popup(
                        'Ошибка',
                        'Порт должен быть целым положительным числом'
                    )
                else:
                    self._window.close()
                    Settings.server_address = (host, port)
                    return True
