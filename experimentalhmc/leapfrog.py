from typing import Union, Callable
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

def leapfrog(position: FloatArray,
             momentum: FloatArray,
             step_size: Union[float, FloatArray],
             steps: int,
             gradient: FloatArray,
             ldg: Callable[[FloatArray, FloatArray], float]):
    ld = 0
    momentum += 0.5 * step_size * gradient

    for step in range(steps):
        position += step_size * momentum
        ld = ldg(position, gradient)
        if step != steps - 1:
            momentum += step_size * gradient

    momentum += 0.5 * step_size * gradient

    return ld
