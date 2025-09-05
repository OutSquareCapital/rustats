import numpy as np
from numpy.typing import NDArray

type _FloatArray = NDArray[np.float64]

def move_sum[T: _FloatArray](
    array: T, length: int, min_length: int, parallel: bool
) -> T:
    """Calculate the moving sum of an array.

    Example:
        >>> import numpy as np
        >>> from rustats import move_sum
        >>> a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> move_sum(a, length=3, min_length=3, parallel=False)
        array([nan, nan,  6.,  9., 12.])
    """
    ...

def move_std[T: _FloatArray](
    array: T, length: int, min_length: int, parallel: bool
) -> T:
    """Calculate the moving standard deviation of an array.

    Example:
        >>> import numpy as np
        >>> from rustats import move_std
        >>> a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> move_std(a, length=3, min_length=3, parallel=False)
        array([nan, nan,  1.,  1.,  1.])
    """
    ...

def move_var[T: _FloatArray](
    array: T, length: int, min_length: int, parallel: bool
) -> T:
    """Calculate the moving variance of an array.

    Example:
        >>> import numpy as np
        >>> from rustats import move_var
        >>> a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> move_var(a, length=3, min_length=3, parallel=False)
        array([nan, nan,  1.,  1.,  1.])
    """
    ...

def move_mean[T: _FloatArray](
    array: T, length: int, min_length: int, parallel: bool
) -> T:
    """Calculate the moving mean of an array.

    Example:
        >>> import numpy as np
        >>> from rustats import move_mean
        >>> a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> move_mean(a, length=3, min_length=3, parallel=False)
        array([nan, nan,  2.,  3.,  4.])
    """
    ...

def move_max[T: _FloatArray](
    array: T, length: int, min_length: int, parallel: bool
) -> T:
    """Calculate the moving maximum of an array.

    Example:
        >>> import numpy as np
        >>> from rustats import move_max
        >>> a = np.array([1.0, 5.0, 2.0, 4.0, 3.0])
        >>> move_max(a, length=3, min_length=3, parallel=False)
        array([nan, nan,  5.,  5.,  4.])
    """
    ...

def move_min[T: _FloatArray](
    array: T, length: int, min_length: int, parallel: bool
) -> T:
    """Calculate the moving minimum of an array.

    Example:
        >>> import numpy as np
        >>> from rustats import move_min
        >>> a = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
        >>> move_min(a, length=3, min_length=3, parallel=False)
        array([nan, nan,  1.,  1.,  2.])
    """
    ...

def move_median[T: _FloatArray](
    array: T, length: int, min_length: int, parallel: bool
) -> T:
    """Calculate the moving median of an array.

    Example:
        >>> import numpy as np
        >>> from rustats import move_median
        >>> a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> move_median(a, length=3, min_length=3, parallel=False)
        array([nan, nan,  2.,  3.,  4.])
    """
    ...

def move_skewness[T: _FloatArray](
    array: T, length: int, min_length: int, parallel: bool
) -> T:
    """Calculate the moving skewness of an array.

    Example:
        >>> import numpy as np
        >>> from rustats import move_skewness
        >>> a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> move_skewness(a, length=3, min_length=3, parallel=False)
        array([nan, nan,  0.,  0.,  0.])
    """
    ...

def move_kurtosis[T: _FloatArray](
    array: T, length: int, min_length: int, parallel: bool
) -> T:
    """Calculate the moving kurtosis of an array.

    Example:
        >>> import numpy as np
        >>> from rustats import move_kurtosis
        >>> a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> move_kurtosis(a, length=3, min_length=3, parallel=False)
        array([nan, nan, -1.5, -1.5, -1.5])
    """
    ...

def move_rank[T: _FloatArray](
    array: T, length: int, min_length: int, parallel: bool
) -> T:
    """Calculate the moving rank of an array.

    Example:
        >>> import numpy as np
        >>> from rustats import move_rank
        >>> a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> move_rank(a, length=3, min_length=3, parallel=False)
        array([nan, nan,  1.,  1.,  1.])
    """
