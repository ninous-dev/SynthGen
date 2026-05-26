import numpy as np
import torch
from noise import pnoise3

def random_perlin_threshold(end: float = 0.4, step: float = 0.05, start: float = 0.0) -> float:
    """
    Returns a biased random float between start and end (inclusive),
    with step size and higher probability for smaller values.

    Parameters
    ----------
    start : float
        Minimum value (default is 0.0).
    end : float
        Maximum value (default is 0.4).
    step : float
        Step size between values (default is 0.05).

    Returns
    -------
    float
        Randomly selected value, biased toward 0.
    """
    values = np.arange(start, end + step, step)
    weights = np.exp(-np.arange(len(values)))
    weights /= weights.sum()

    return np.random.choice(values, p=weights)

def generate_perlin_map(
    shape: tuple[int, int, int],
    scale: float = 20.0,
    octaves: int = 4,
    lacunarity: float = 2.0,
    base: int = 0
) -> torch.Tensor:
    """
    Generate a Perlin noise map.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Output shape (Z, Y, X).
    scale : float, optional
        Base frequency/scale of the noise, by default=20.0.
    octaves : int, optional
        Number of octaves, by default=4.
    lacunarity : float, optional
        Frequency multiplier between octaves, by default=2.0.
    base : int, optional
        Seed for noise generation, by default=0.

    Returns
    -------
    torch.Tensor
        Noise volume with shape `shape`.
    """
    z, y, x = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij'
    )

    # Vectorize version of pnoise3
    pnoise3_vec = np.vectorize(lambda x, y, z: pnoise3(
        x / scale, y / scale, z / scale,
        octaves=octaves,
        lacunarity=lacunarity,
        repeatx=1024,
        repeaty=1024,
        repeatz=1024,
        base=base
    ), otypes=[np.float32])

    return pnoise3_vec(x, y, z)