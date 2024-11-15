import PySimpleGUI as sg

from mai.functions import popup

__all__ = ('HostPort',)

class HostPort:
    __slots__ = (
        '_window',
        '_default_host', 'host',
        '_default_port', 'port'
    )
    _window: sg.Window
    _default_host: str
    _default_port: int

    def __init__(
        self,
        default_host: str = 'localhost',
        default_port: int = 11545
    ):
        assert isinstance(default_host, str)
        assert isinstance(default_port, int)
        assert default_port > 0
        self._default_host = default_host
        self._default_port = default_port
        self.host = None
        self.port = None

    def _build_window(self) -> None:
        host = self._default_host if self.host is None else self.host
        port = self._default_port if self.port is None else self.port
        self._window = sg.Window('MAI', [
            [sg.Text('Enter Rocket League host:port', pad=(0, 0, 0, 5), font=('Comic Sans', 10))],
            [
                sg.InputText(default_text=host, focus=True, size=(25, 10), pad=0, border_width=0),
                sg.Text(':', pad=0),
                sg.InputText(default_text=str(port), size=(5, 10), pad=0, border_width=0)
            ],
            [sg.Submit(pad=(0, 5, 0, 0), font=('Comic Sans', 10))]
        ])

    def run(self):
        self._build_window()
        while True:
            event, values = self._window.read()
            values: dict

            if event == sg.WIN_CLOSED:
                break

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
                    return
