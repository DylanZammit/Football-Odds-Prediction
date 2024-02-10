import numpy as np
import os
import pickle


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


def save(obj, model_name_pkl: str):
    """
    Save an object as a pickle for reuse
    """
    base_dir = os.path.dirname(__file__)
    model_name_pkl = os.path.join(base_dir, model_name_pkl)
    try:
        with open(model_name_pkl, 'wb+') as f:
            print(f'Saving to {model_name_pkl}...', end='', flush=True)
            pickle.dump(obj, f)
    except Exception as e:
        print(f'FAILED TO SAVE: {e}')
    else:
        print(f'saved!')


def load(model_name_pkl: str):
    """
    Load and return a pickle object
    """
    base_dir = os.path.dirname(__file__)
    model_name_pkl = os.path.join(base_dir, model_name_pkl)
    with open(model_name_pkl, 'rb') as f:
        print(f'loading {model_name_pkl}...', end='', flush=True)
        model = pickle.load(f)
    print(f'loaded {model_name_pkl}')
    return model

