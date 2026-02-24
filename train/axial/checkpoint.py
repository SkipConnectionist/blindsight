import os
import socket
import torch
import logging

from datetime import datetime
from pathlib import Path
from typing import Any
from collections import deque


logger = logging.getLogger('Checkpoint')


class Checkpointer:
    """
    A PyTorch checkpointer that saves checkpoints and manages cleanup.

    Checkpoint naming format: {prefix}_{timestamp}_{hostname}_step_{step}.pt
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        prefix: str = 'checkpoint',
        max_checkpoints: int | None = None
    ):
        """
        Initialize the checkpointer.

        Args:
            checkpoint_dir: Directory to save checkpoints
            prefix: Prefix for checkpoint filenames
            max_checkpoints: Maximum number of checkpoints to keep (None for unlimited)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.prefix = prefix
        self.max_checkpoints = max_checkpoints
        self.hostname = socket.gethostname().removesuffix('.local')

        # Keep track of checkpoints saved by this instance
        self.saved_checkpoints: deque = deque()

        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self, step: int) -> str:
        """
        Generate checkpoint filename with all required components.
        """
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
        filename = f"{self.prefix}_{timestamp}_{self.hostname}_step_{step}.pt"
        return filename

    def save(
        self,
        state_dict: dict[str, Any],
        step_index: int,
        epoch_index: int,
        optimizer_state: dict[str, Any] | None = None,
        scheduler_state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> Path:
        """
        Save a checkpoint with the given state.

        Args:
            state_dict: Model state dictionary
            step: Current training step
            optimizer_state: Optimizer state dictionary (optional)
            scheduler_state: Scheduler state dictionary (optional)
            metadata: Additional metadata to save (optional)

        Returns:
            Path to the saved checkpoint file
        """
        filename = self._generate_filename(step_index + 1)
        filepath = self.checkpoint_dir / filename

        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': state_dict,
            'step_index': step_index,
            'epoch_index': epoch_index,
            'timestamp': datetime.now().isoformat(),
            'hostname': self.hostname,
        }

        # Add optional components
        if optimizer_state is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer_state

        if scheduler_state is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler_state

        if metadata is not None:
            checkpoint_data['metadata'] = metadata

        # Save checkpoint
        torch.save(checkpoint_data, filepath)

        # Track this checkpoint
        self.saved_checkpoints.append(filepath)

        # Clean up old checkpoints if necessary
        self._cleanup_old_checkpoints()

        logger.info(f"Checkpoint saved: {filepath}")

        return filepath

    def _cleanup_old_checkpoints(self):
        """
        Remove oldest checkpoints if max_checkpoints limit is exceeded.
        """
        if self.max_checkpoints is None:
            return

        while len(self.saved_checkpoints) > self.max_checkpoints:
            old_checkpoint = self.saved_checkpoints.popleft()
            try:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f'Deleted old checkpoint: {old_checkpoint}')
            except OSError as e:
                logging.warning(f'Failed to remove checkpoint {old_checkpoint}: {e}')

    def get_latest_checkpoint_path(self) -> Path | None:
        """
        Get the path to the most recently saved checkpoint by this instance.

        Returns:
            Path to latest checkpoint or None if no checkpoints saved
        """
        if not self.saved_checkpoints:
            return None
        return self.saved_checkpoints[-1]

    def get_saved_checkpoint_paths(self) -> list[Path]:
        """
        List all checkpoints saved by this instance.

        Returns:
            List of checkpoint file paths
        """
        return list(self.saved_checkpoints)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> dict[str, Any]:
        """
        Load a checkpoint from the given path.

        Args:
            path: Path to checkpoint file

        Returns:
            Loaded checkpoint data
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location='cpu')
        logger.info(f'Checkpoint loaded: {path}')
        return checkpoint
    
    @classmethod
    def restore(
        cls,
        path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> dict[str, Any]:
        checkpoint = cls.load_checkpoint(path)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer_state = checkpoint.get('optimizer_state_dict')
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            else:
                logger.warning('No optimizer state found in checkpoint.')

        if scheduler is not None:
            scheduler_state = checkpoint.get('scheduler_state_dict')
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            else:
                logger.warning('No scheduler state found in checkpoint.')

        return checkpoint

    @classmethod
    def make_checkpoint_dir(
        cls,
        root: str,
        prefix: str,
    ):
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        name = f'{prefix}_{timestamp}'
        path = Path(root) / name
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f'Created checkpoint directory: {path}')
        return path
