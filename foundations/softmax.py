import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z = z - np.max(z)          # Numerical stability
        exp_z = np.exp(z)          # Exponentials
        softmax = exp_z / np.sum(exp_z)  # Normalize
        return np.round(softmax, 4)