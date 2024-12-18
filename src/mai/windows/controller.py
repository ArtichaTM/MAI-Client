from typing import Callable
from threading import Thread

import win32gui as wgui
import win32con as wcon
import win32api as wapi


def find_window(checker: Callable[[str], bool]) -> int | None:
    handle: int | None = None
    def _window_enum_callback(hwnd: int, checker: Callable[[str], bool]):
        nonlocal handle
        """Pass to win32gui.EnumWindows() to check all the opened windows"""
        if checker(wgui.GetWindowText(hwnd)):
            handle = hwnd
    wgui.EnumWindows(_window_enum_callback, checker)
    return handle


class WindowController:
    _instance: 'WindowController | None' = None
    __slots__ = ('_hwnd',)
    _hwnd: int | None

    def __init__(self) -> None:
        assert self._instance is None
        type(self)._instance = self
        self.update_window()

    @staticmethod
    def check_rl_name(name: str) -> bool:
        return name.startswith('Rocket League (') and name.endswith(')')

    @property
    def window_exists(self) -> bool:
        return self._hwnd is not None

    def update_window(self) -> None:
        self._hwnd = find_window(self.check_rl_name)

    def _press_key_int(self, key: int, hold_time: float = 0.0) -> None:
        assert self._hwnd is not None
        assert isinstance(key, int)
        wapi.PostMessage(self._hwnd, wcon.WM_KEYDOWN, key, 0)
        
        wapi.PostMessage(self._hwnd, wcon.WM_KEYUP, key, 0)

    def _press_key_str(self, key: str, hold_time: float = 0.0) -> None:
        assert self._hwnd is not None
        assert isinstance(key, str)
        raise NotImplementedError()

    def press_key(self, key: int | str, hold_time: float = 0.0) -> None:
        assert self._hwnd is not None
        if isinstance(key, int):
            return self._press_key_int(key, hold_time=hold_time)
        if isinstance(key, str):
            return self._press_key_str(key, hold_time=hold_time)
