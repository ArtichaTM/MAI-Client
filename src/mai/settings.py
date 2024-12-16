from pathlib import Path

__all__ = ("Settings",)

class _Settings(dict):
    __slots__ = (
        'server_address', 'timeout_seconds',
        'max_speed_magnitude',

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
    button_restart_training: int
    button_skip_replay: int

    def __init__(self) -> None:
        self.server_address = ('localhost', 11545)
        self.timeout_seconds = 2.0
        self.max_speed_magnitude = 0.32
        self.button_restart_training = 5
        self.button_skip_replay = 2

Settings = _Settings()
