import numpy as np


def obs2tensor(obs: dict) -> np.array:
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
    return None


def _get_halite(obs: dict) -> np.array:
    """
    Convert halite list into 2D tensor.

    Args:
        observation (dict): [description]

    Returns:
        np.array: (21, 21) shaped array in 0-1 range.
    """
    shape = _get_shape(obs)
    return np.array(obs["observation"]["halite"]).reshape(shape, shape) / 500


def _get_players_state(obs: dict) -> np.array:
    pass


def _get_shape(obs: dict) -> int:
    """
    Get the shape of a board.

    Args:
        observation (dict): [description]

    Returns:
        int: shape
    """
    shape = len(obs["observation"]["halite"]) ** 0.5
    assert shape == int(shape)
    return int(shape)


def _get_player_no(state: list) -> int:
    """
    Get the number of players.

    Args:
        state (list): [description]

    Returns:
        int: players number
    """
    return len(state)
