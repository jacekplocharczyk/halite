import numpy as np


def obs2tensor(observation: dict) -> np.array:
    """
    Convert observation from the envionment into 3D tensor.

    Args:
        observation (dict): [description]

    Returns:
        np.array: (13, 21, 21) shaped tensor. Descriptions:
            [0, :, :] - halite on the map
            [1, :, :] - player #1 ships
            [2, :, :] - player #1 halite on the ships
            [3, :, :] - player #1 shipsyards
            [4:7, :, :] - player #2
            [7:10, :, :] - player #3
            [10:13, :, :] - player #4

    """
    del observation
    return None


def _get_halite(observation: dict) -> np.array:
    """
    Convert halite list into 2D tensor.

    Args:
        observation (dict): [description]

    Returns:
        np.array: (21, 21) shaped array in 0-1 range.
    """
    shape = len(observation["observation"]["halite"]) ** 0.5
    assert shape == int(shape)
    shape = int(shape)
    return np.array(observation["observation"]["halite"]).reshape(shape, shape) / 500

