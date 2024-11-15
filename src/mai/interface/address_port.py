import PySimpleGUI as sg

from mai.functions import popup

__all__ = ('AddressPort',)

class AddressPort:
    __slots__ = (
        '_window',
        '_default_address', 'address',
        '_default_port', 'port'
    )
    _window: sg.Window
    _default_address: str
    _default_port: int

    def __init__(
        self,
        default_address: str = 'localhost',
        default_port: int = 11545
    ):
        assert isinstance(default_address, str)
        assert isinstance(default_port, int)
        assert default_port > 0
        self._default_address = default_address
        self._default_port = default_port
        self.address = None
        self.port = None

    def _build_window(self) -> None:
        address = self._default_address if self.address is None else self.address
        port = self._default_port if self.port is None else self.port
        self._window = sg.Window('MAI', [
            [sg.Text('Enter Rocket League address:port', pad=(0, 0, 0, 5), font=('Comic Sans', 10))],
            [
                sg.InputText(default_text=address, focus=True, size=(25, 10), pad=0, border_width=0),
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
                address, port = values.values()
                self.address = address
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
