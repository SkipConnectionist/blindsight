import torch
import random

from typing import Callable, Iterator
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils._pytree import tree_map

from axial.data.typing import Randomizable, Indexable, SizedIterable


class DataPipeline(Dataset):

    def __init__(
        self,
        dataset,
        *transforms: Callable
    ):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Shallow copy before transforming
        item = dict(item)

        # Transform
        for transform in self.transforms:
            item = transform(item)

        return item


class MapToDevice(SizedIterable):

    def __init__(
        self,
        source: SizedIterable,
        device: str | torch.device
    ):
        self.source = source
        self.device = device

    def __iter__(self) -> Iterator:
        device = self.device
        map_to_device = lambda x: x.to(device) if torch.is_tensor(x) else x
        for data in self.source:
            yield tree_map(map_to_device, data)

    def __len__(self):
        return len(self.source)


def randomize(dataset):
    if isinstance(dataset, Randomizable):
        dataset.randomize()


class RandomizingConcat(ConcatDataset, Dataset):

    def randomize(self):
        for dataset in self.datasets:
            randomize(dataset)


class RandomSubset[T](Dataset, Randomizable):
    def __init__(self, dataset: Indexable[T], size: int):
        assert len(dataset) >= size
        self.dataset = dataset
        self.size = size
        self.lut = self._sample_indices()

    def _sample_indices(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        return indices[:self.size]

    def __len__(self):
        return self.size

    def __getitem__(self, index) -> T:
        return self.dataset[self.lut[index]]

    def randomize(self):
        self.lut = self._sample_indices()
        randomize(self.dataset)


class RandomizingDataLoader[T](SizedIterable):

    def __init__(self, dataloader: DataLoader[T]):
        self.dataloader = dataloader

    def __iter__(self) -> Iterator[T]:
        for batch in self.dataloader:
            yield batch
        randomize(self.dataloader.dataset)

    def __len__(self):
        return len(self.dataloader)


def batch_sequences_with_padding(
    sequences: list[torch.Tensor],
    pad_value: float = 0.0,
    pad_to_length: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batch a list of variable-length sequences into a single padded tensor with attention masks.

    Args:
        sequences: List of tensors, each of shape (S_i, N) where S_i varies (sequence_len, features)
        pad_value: Value to use for padding (default: 0.0)
        pad_to_length: Optional fixed length to pad to. If None, pads to max length in batch

    Returns:
        Tuple of:
        - batched_tensor: Shape (B, S_max, N) where B=len(sequences), S_max is the maximum sequence length
        - attention_mask: Shape (B, S_max) where 1=real token, 0=padded token
    """
    if not sequences:
        raise ValueError("Cannot batch empty list of sequences")

    # Get dimensions
    batch_size = len(sequences)
    N = sequences[0].shape[1]  # Feature dimension (should be same for all)

    # Validate that all sequences have the same N dimension
    for i, seq in enumerate(sequences):
        if len(seq.shape) != 2:
            raise ValueError(f"Expected 2D tensor at index {i}, got shape {seq.shape}")
        if seq.shape[1] != N:
            raise ValueError(f"All sequences must have same feature dimension. "
                             f"Expected {N}, got {seq.shape[1]} at index {i}")

    # Get sequence lengths
    seq_lengths = [seq.shape[0] for seq in sequences]

    # Determine target length
    if pad_to_length is not None:
        max_length = pad_to_length
        if max_length < max(seq_lengths):
            raise ValueError(f"pad_to_length ({pad_to_length}) is smaller than "
                             f"maximum sequence length ({max(seq_lengths)})")
    else:
        max_length = max(seq_lengths)

    # Get device and dtype from first sequence
    device = sequences[0].device
    dtype = sequences[0].dtype

    # Initialize batched tensor with padding value
    batched_tensor = torch.full(
        (batch_size, max_length, N),
        pad_value,
        dtype=dtype,
        device=device
    )

    # Initialize attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long,
        device=device
    )

    # Fill in the sequences and create masks
    for i, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        batched_tensor[i, :seq_len, :] = seq
        attention_mask[i, :seq_len] = 1

    return batched_tensor, attention_mask
