__all__ = ("Settings",)

class _Settings(dict):
    __slots__ = ('server_address',)
    server_address: tuple[str, int]

Settings = _Settings()
