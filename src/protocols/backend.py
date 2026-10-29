"""Backend type protocol for NumPy/CuPy compatibility."""

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

#: Array type shared by the numpy and cupy backends. numpy is the typed side of
#: the pair (cupy ships no `py.typed`), so it is what callers are annotated
#: against and what they receive.
type Array = NDArray[Any]

#: A CuPy array. Deliberately `Any`: CuPy has no usable type information, so the
#: only honest spelling is "unchecked". Naming it keeps `-> np.ndarray` out of
#: the GPU accessors (a false claim) and gives one place to change when CuPy is
#: typed.
type CupyArray = Any


class BackendModule(Protocol):
  """Protocol for array backend modules (numpy or cupy).

  Defines the common array operations across NumPy and CuPy used by
  the modular solver without generic type parameter explosion::

      def build(xp: BackendModule = np) -> Array: ...

  Only members that exist on both backends are declared. Reductions that may
  return either an array or a scalar (`sum`, `min`, `max`, `argmin`, `argmax`)
  stay `Any` on purpose: their true return type depends on the arguments.
  """

  # --- constants and dtype aliases -----------------------------------------
  # Declared as read-only properties: a plain attribute declaration in a
  # protocol is writable, and a module attribute cannot satisfy that.
  @property
  def newaxis(self) -> None:
    """Axis-insertion constant."""
    ...

  @property
  def ndarray(self) -> type[Array]:
    """Array class of the backend."""
    ...

  @property
  def int32(self) -> type[np.int32]:
    """32-bit integer dtype (numpy's, re-exported by cupy)."""
    ...

  @property
  def int64(self) -> type[np.int64]:
    """64-bit integer dtype (numpy's, re-exported by cupy)."""
    ...

  @property
  def random(self) -> Any:
    """Random number generation module."""
    ...

  # --- constructors --------------------------------------------------------
  def array(self, *args: Any, **kwargs: Any) -> Array:
    """Create an array."""
    ...

  def asarray(self, *args: Any, **kwargs: Any) -> Array:
    """Convert to array."""
    ...

  def zeros(self, *args: Any, **kwargs: Any) -> Array:
    """Create array of zeros."""
    ...

  def ones(self, *args: Any, **kwargs: Any) -> Array:
    """Create array of ones."""
    ...

  def empty(self, *args: Any, **kwargs: Any) -> Array:
    """Create empty array."""
    ...

  def arange(self, *args: Any, **kwargs: Any) -> Array:
    """Create range array."""
    ...

  def vstack(self, *args: Any, **kwargs: Any) -> Array:
    """Stack arrays vertically."""
    ...

  def concatenate(self, *args: Any, **kwargs: Any) -> Array:
    """Join arrays along an existing axis."""
    ...

  def where(self, *args: Any, **kwargs: Any) -> Array:
    """Return elements chosen from two arrays by a condition."""
    ...

  def clip(self, *args: Any, **kwargs: Any) -> Array:
    """Limit values to an interval."""
    ...

  def argsort(self, *args: Any, **kwargs: Any) -> Array:
    """Return indices that would sort an array."""
    ...

  # --- element-wise math ---------------------------------------------------
  def abs(self, *args: Any, **kwargs: Any) -> Array:
    """Absolute value."""
    ...

  def ceil(self, *args: Any, **kwargs: Any) -> Array:
    """Ceiling."""
    ...

  def cos(self, *args: Any, **kwargs: Any) -> Array:
    """Cosine."""
    ...

  def sin(self, *args: Any, **kwargs: Any) -> Array:
    """Sine."""
    ...

  def arccos(self, *args: Any, **kwargs: Any) -> Array:
    """Inverse cosine."""
    ...

  def exp(self, *args: Any, **kwargs: Any) -> Array:
    """Exponential."""
    ...

  def sqrt(self, *args: Any, **kwargs: Any) -> Array:
    """Square root."""
    ...

  def trunc(self, *args: Any, **kwargs: Any) -> Array:
    """Truncate toward zero."""
    ...

  # --- in-place / scalar-returning -----------------------------------------
  def fill_diagonal(self, *args: Any, **kwargs: Any) -> None:
    """Fill the main diagonal of a 2-D array in place."""
    ...

  def sum(self, *args: Any, **kwargs: Any) -> Any:
    """Sum array elements (scalar or array depending on arguments)."""
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
