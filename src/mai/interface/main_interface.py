import PySimpleGUI as sg
from mai.capnp import CapnPClient

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
                [sg.Text('Overview')],
                [sg.Button('Update')]
            ]),
            sg.Tab('Stats', [
                [sg.Text('Stats')]
            ])
        ]], expand_x=True, expand_y=True)]], margins=(0, 0))

    def run(self, broker: CapnPClient):
        while True:
            event, values = self._window.read()
            values: dict

            if event == sg.WIN_CLOSED:
                break

            if event == 'Update':
                print('Updating...')
                controls = broker.controls_type.new_message()
                controls.throttle = 0
                controls.steer = 0
                controls.pitch = 0
                controls.yaw = 0
                controls.roll = 0
                controls.boost = False
                controls.jump = False
                data = broker.exchange(controls)
                print('Received data:', data)
