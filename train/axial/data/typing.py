from typing import Iterable, Protocol, runtime_checkable
from collections.abc import Sized, Iterable


@runtime_checkable
class Randomizable(Protocol):

    def randomize(self):
        ...


class Indexable[T](Sized, Protocol):

    def __getitem__(self, index: int) -> T:
        ...


class SizedIterable(Sized, Iterable, Protocol):
    ...
