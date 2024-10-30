from typing import Callable
from sys import argv

from .settings import Colors
from . import socketing


def run(args: list[str]) -> None:
    """
    Verifies loaded voice lines
    and loads the missing ones
    """
    print("Running...")


def help(args: list[str]) -> None:
    """
    Prints help docstring from function. Example:
    > help help
    """
    if not args:
        args = ['help']

    func = FUNCTIONS.get(args[0])
    if func is None:
        print(f"{Colors.FAIL}Unknown Command{Colors.END}: {Colors.BOLD}{args[0]}{Colors.END}")
        return
    if func.__doc__ is None:
        print(f"Command {Colors.BOLD}{args[0]}{Colors.END} {Colors.FAIL}has no{Colors.END} help string)")
    print(func.__doc__)


FUNCTIONS: dict[str, Callable[[list[str],], Callable]] = {
    'run': run,
    'help': help,
}


def main():
    args = argv[1:]
    if len(args) == 0:
        print(f"{Colors.FAIL}Exception:{Colors.END} First argument is mandatory")
        return
    func = FUNCTIONS.get(args[0])
    if func is None:
        print(f"{Colors.FAIL}Unknown Command{Colors.END}: {Colors.BOLD}{args[0]}{Colors.END}")
        print(f"{Colors.GREEN}Available{Colors.END}: {Colors.BOLD}", end='')
        print(f'{Colors.END}, {Colors.BOLD}'.join(FUNCTIONS.keys()) + Colors.END)
        return
    try:
        func(args[1:])
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
