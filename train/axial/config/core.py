from typing import TypedDict, Protocol, Callable, NotRequired
from functools import cache
from pathlib import Path

from axial.data.typing import SizedIterable

import torch

type Creator[T] = Callable[[], T]


class OptimizerCreator(Protocol):

    def __call__(
        self,
        params: list[torch.nn.Parameter]
    ) -> torch.optim.Optimizer:
        ...


class SchedulerCreator(Protocol):

    def __call__(
        self,
        optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        ...


class LossFunction(Protocol):

    def __call__(self, output, batch) -> torch.Tensor:
        ...


class Config(TypedDict):
    model: Creator[torch.nn.Module]
    loss_fn: Creator[LossFunction]
    training_data: Creator[SizedIterable]
    validation_data: Creator[SizedIterable]
    optimizer: OptimizerCreator
    scheduler: NotRequired[SchedulerCreator]
    gradient_accumulation_steps: NotRequired[int]
    max_gradient_norm: NotRequired[float]
    num_epochs: NotRequired[int]
    model_initializer: NotRequired[Callable[[torch.nn.Module], None]]


class ConfigSource(Protocol):

    def __call__(self) -> Config:
        ...


configs: dict[str, ConfigSource] = {}


def get_host_data_path(path: str) -> Path:
    return Path.home() / 'data' / path


def register(source: ConfigSource) -> ConfigSource:
    name = getattr(source, '__name__', source.__class__.__name__)
    assert name not in configs, f'Config "{name}" already registered'
    configs[name] = source
    return source


def with_cached_creators(config: Config) -> Config:
    config = Config(**config)

    for key, value in config.items():
        if callable(value):
            config[key] = cache(value)

    return config
