"""This is an automatically generated stub for `data.capnp`."""

from __future__ import annotations

from contextlib import contextmanager
from io import BufferedWriter
from typing import Iterator, Literal, overload

class MAIVector:
    x: float
    y: float
    z: float
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes, traversal_limit_in_words: int | None = ..., nesting_limit: int | None = ...
    ) -> Iterator[MAIVectorReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes, traversal_limit_in_words: int | None = ..., nesting_limit: int | None = ...
    ) -> MAIVectorReader: ...
    @staticmethod
    def new_message() -> MAIVectorBuilder: ...
    def to_dict(self) -> dict: ...

class MAIVectorReader(MAIVector):
    def as_builder(self) -> MAIVectorBuilder: ...

class MAIVectorBuilder(MAIVector):
    @staticmethod
    def from_dict(dictionary: dict) -> MAIVectorBuilder: ...
    def copy(self) -> MAIVectorBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> MAIVectorReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...

class MAIRotator:
    pitch: int
    roll: int
    yaw: int
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes, traversal_limit_in_words: int | None = ..., nesting_limit: int | None = ...
    ) -> Iterator[MAIRotatorReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes, traversal_limit_in_words: int | None = ..., nesting_limit: int | None = ...
    ) -> MAIRotatorReader: ...
    @staticmethod
    def new_message() -> MAIRotatorBuilder: ...
    def to_dict(self) -> dict: ...

class MAIRotatorReader(MAIRotator):
    def as_builder(self) -> MAIRotatorBuilder: ...

class MAIRotatorBuilder(MAIRotator):
    @staticmethod
    def from_dict(dictionary: dict) -> MAIRotatorBuilder: ...
    def copy(self) -> MAIRotatorBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> MAIRotatorReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...

class MAIRLObjectState:
    position: MAIVector | MAIVectorBuilder | MAIVectorReader
    velocity: MAIVector | MAIVectorBuilder | MAIVectorReader
    rotation: MAIRotator | MAIRotatorBuilder | MAIRotatorReader
    angularVelocity: MAIVector | MAIVectorBuilder | MAIVectorReader
    @overload
    def init(self, name: Literal["position"]) -> MAIVector: ...
    @overload
    def init(self, name: Literal["velocity"]) -> MAIVector: ...
    @overload
    def init(self, name: Literal["rotation"]) -> MAIRotator: ...
    @overload
    def init(self, name: Literal["angularVelocity"]) -> MAIVector: ...
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes, traversal_limit_in_words: int | None = ..., nesting_limit: int | None = ...
    ) -> Iterator[MAIRLObjectStateReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes, traversal_limit_in_words: int | None = ..., nesting_limit: int | None = ...
    ) -> MAIRLObjectStateReader: ...
    @staticmethod
    def new_message() -> MAIRLObjectStateBuilder: ...
    def to_dict(self) -> dict: ...

class MAIRLObjectStateReader(MAIRLObjectState):
    position: MAIVectorReader
    velocity: MAIVectorReader
    rotation: MAIRotatorReader
    angularVelocity: MAIVectorReader
    def as_builder(self) -> MAIRLObjectStateBuilder: ...

class MAIRLObjectStateBuilder(MAIRLObjectState):
    position: MAIVector | MAIVectorBuilder | MAIVectorReader
    velocity: MAIVector | MAIVectorBuilder | MAIVectorReader
    rotation: MAIRotator | MAIRotatorBuilder | MAIRotatorReader
    angularVelocity: MAIVector | MAIVectorBuilder | MAIVectorReader
    @staticmethod
    def from_dict(dictionary: dict) -> MAIRLObjectStateBuilder: ...
    def copy(self) -> MAIRLObjectStateBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> MAIRLObjectStateReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...

class MAIGameState:
    car: MAIRLObjectState | MAIRLObjectStateBuilder | MAIRLObjectStateReader
    ball: MAIRLObjectState | MAIRLObjectStateBuilder | MAIRLObjectStateReader
    boostAmount: float
    dead: bool
    @overload
    def init(self, name: Literal["car"]) -> MAIRLObjectState: ...
    @overload
    def init(self, name: Literal["ball"]) -> MAIRLObjectState: ...
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes, traversal_limit_in_words: int | None = ..., nesting_limit: int | None = ...
    ) -> Iterator[MAIGameStateReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes, traversal_limit_in_words: int | None = ..., nesting_limit: int | None = ...
    ) -> MAIGameStateReader: ...
    @staticmethod
    def new_message() -> MAIGameStateBuilder: ...
    def to_dict(self) -> dict: ...

class MAIGameStateReader(MAIGameState):
    car: MAIRLObjectStateReader
    ball: MAIRLObjectStateReader
    def as_builder(self) -> MAIGameStateBuilder: ...

class MAIGameStateBuilder(MAIGameState):
    car: MAIRLObjectState | MAIRLObjectStateBuilder | MAIRLObjectStateReader
    ball: MAIRLObjectState | MAIRLObjectStateBuilder | MAIRLObjectStateReader
    @staticmethod
    def from_dict(dictionary: dict) -> MAIGameStateBuilder: ...
    def copy(self) -> MAIGameStateBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> MAIGameStateReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...

class MAIControls:
    throttle: float
    steer: float
    pitch: float
    yaw: float
    roll: float
    boost: bool
    jump: bool
    handbrake: bool
    dodgeForward: int
    dodgeStrafe: int
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes, traversal_limit_in_words: int | None = ..., nesting_limit: int | None = ...
    ) -> Iterator[MAIControlsReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes, traversal_limit_in_words: int | None = ..., nesting_limit: int | None = ...
    ) -> MAIControlsReader: ...
    @staticmethod
    def new_message() -> MAIControlsBuilder: ...
    def to_dict(self) -> dict: ...

class MAIControlsReader(MAIControls):
    def as_builder(self) -> MAIControlsBuilder: ...

class MAIControlsBuilder(MAIControls):
    @staticmethod
    def from_dict(dictionary: dict) -> MAIControlsBuilder: ...
    def copy(self) -> MAIControlsBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> MAIControlsReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...
