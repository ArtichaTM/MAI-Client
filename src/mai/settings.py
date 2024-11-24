from pathlib import Path

__all__ = ("Settings",)

class _Settings(dict):
    __slots__ = ('server_address','timeout_seconds')
    path_to_capnp_schemes = Path = (
        Path(__file__)
        .parent.parent.parent.parent
        .joinpath('capnp').absolute()
    )
    server_address: tuple[str, int]
    timeout_seconds: float  # Seconds

    def __init__(self) -> None:
        self.server_address = ('localhost', 11545)
        self.timeout_seconds = 2.0

Settings = _Settings()
