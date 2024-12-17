from pathlib import Path
from enum import IntEnum

__all__ = ('WinButtons', 'Settings')

class WinButtons(IntEnum):
    SKIP_REPLAY = 2
    RESTART_TRAINING = 5
    FORWARD = 87


class _Settings(dict):
    __slots__ = (
        'server_address', 'timeout_seconds',
        'max_speed_magnitude', 'control_apply_threshold',

        # Buttons
        'button_restart_training', 'button_skip_replay'
    )
    path_to_capnp_schemes = Path = (
        Path(__file__)
        .parent.parent.parent.parent
        .joinpath('capnp').absolute()
    )
    server_address: tuple[str, int]
    timeout_seconds: float  # Seconds
    max_speed_magnitude: float
    control_apply_threshold: float

    def __init__(self) -> None:
        self.server_address = ('localhost', 11545)
        self.timeout_seconds = 2.0
        self.max_speed_magnitude = 0.32
        self.control_apply_threshold = 1e-06

Settings = _Settings()
