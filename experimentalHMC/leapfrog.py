from typing import Union, Callable
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

def leapfrog(position: FloatArray,
             momentum: FloatArray,
             stepsize: Union[float, FloatArray],
             steps: int,
             gradient: FloatArray,
             ldg: Callable[[FloatArray, FloatArray], float]):
    ld = 0
    momentum += 0.5 * stepsize * gradient

    for step in range(steps):
        position += stepsize * momentum
        ld = ldg(position, gradient)
        if step != steps - 1:
            momentum += stepsize * gradient

    momentum += 0.5 * stepsize * gradient

    return ld
