from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class MovingLeastSquares(ABC):
    """
    Moving Least Squares (MLS) smoother for 3D point clouds.
    """

    @abstractmethod
    def run(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run MLS on all points and return (proj, normals or None).
        """
        pass

    @abstractmethod
    def project_points(self, i: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Project point P[i] onto locally fitted surface; optionally return normal.
        """
        pass
