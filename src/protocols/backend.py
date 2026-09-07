"""Backend type protocol for NumPy/CuPy compatibility."""

from typing import Protocol, Any


class BackendModule(Protocol):
    """Protocol for array backend modules (numpy or cupy).

    This allows type-safe backend switching between NumPy (CPU)
    and CuPy (GPU) without runtime overhead.
    """

    def array(self, *args: Any, **kwargs: Any) -> Any:
        """Create an array."""
        ...

    def asarray(self, *args: Any, **kwargs: Any) -> Any:
        """Convert to array."""
        ...

    def zeros(self, *args: Any, **kwargs: Any) -> Any:
        """Create array of zeros."""
        ...

    def ones(self, *args: Any, **kwargs: Any) -> Any:
        """Create array of ones."""
        ...

    def empty(self, *args: Any, **kwargs: Any) -> Any:
        """Create empty array."""
        ...

    def arange(self, *args: Any, **kwargs: Any) -> Any:
        """Create range array."""
        ...

    def sum(self, *args: Any, **kwargs: Any) -> Any:
        """Sum array elements."""
        ...

    def min(self, *args: Any, **kwargs: Any) -> Any:
        """Find minimum."""
        ...

    def max(self, *args: Any, **kwargs: Any) -> Any:
        """Find maximum."""
        ...

    def argmin(self, *args: Any, **kwargs: Any) -> Any:
        """Find index of minimum."""
        ...

    def argmax(self, *args: Any, **kwargs: Any) -> Any:
        """Find index of maximum."""
        ...

    def sqrt(self, *args: Any, **kwargs: Any) -> Any:
        """Square root."""
        ...

    def exp(self, *args: Any, **kwargs: Any) -> Any:
        """Exponential."""
        ...

    def random(self) -> Any:
        """Random number generation module."""
        ...
