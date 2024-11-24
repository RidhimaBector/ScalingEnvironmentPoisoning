from abc import ABC, abstractmethod
from typing import Any, List, Tuple

class Buffer(ABC):
    """Abstract base class for experience replay buffers."""

    @abstractmethod
    def push(self, experience: Tuple[Any, ...]) -> None:
        """Add experience to buffer."""
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> List[Tuple[Any, ...]]:
        """Sample batch of experiences."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return current buffer size."""
        pass
