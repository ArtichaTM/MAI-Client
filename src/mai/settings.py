from pathlib import Path

__all__ = ("Settings",)

class _Settings(dict):
    __slots__ = ('server_address',)
    path_to_capnp_schemes = Path = (
        Path(__file__)
        .parent.parent.parent.parent
        .joinpath('capnp').absolute()
    )
    server_address: tuple[str, int]

    def __init__(self) -> None:
        self.server_address = ('localhost', 11545)

Settings = _Settings()
