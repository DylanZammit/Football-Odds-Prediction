import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute().parent.absolute().parent.absolute()


def expon(x: float, lam: float):
    """
    Un-normalised exponential distribution
    """
    return lam ** x * np.exp(-lam)


def decay(x: float, zeta: float = 0.002) -> float:
    """
    Exponential decay used to weigh down old matches in likelihood
    """
    return np.exp(-zeta * x)


if __name__ == '__main__':
    print(ROOT_DIR)
