from typing import Tuple

import numpy as np


def obs2tensor(obs: dict) -> np.array:
    """
    Convert observation from the envionment into 3D tensor.

    Args:
        observation (dict): dictionary with observations from the Board object.

    Returns:
        np.array: (13, 21, 21) shaped tensor. Descriptions:
            [0, :, :] - halite on the map
            [1, :, :] - player #1 ships
            [2, :, :] - player #1 halite on the ships
            [3, :, :] - player #1 shipyards
            [4:7, :, :] - player #2
            [7:10, :, :] - player #3
            [10:13, :, :] - player #4

    """
    shape = _get_shape(obs)
    players_no = _get_player_no(obs)

    tensor = np.empty((3 * players_no + 1, shape, shape))
    tensor[0] = _get_halite(obs, shape)
    tensor[1:] = _get_players_state(obs, shape, players_no)
    return tensor


def _get_halite(obs: dict, shape: int) -> np.array:
    """
    Convert halite list into 2D tensor.

    Args:
        observation (dict): dictionary with observations from the Board object.
        shape (int): Board shape

    Returns:
        np.array: (21, 21) shaped array in 0-1 range.
    """
    return np.array(obs["halite"]).reshape(shape, shape) / 500


def _get_players_state(obs: dict, shape: int, players_no: int) -> np.array:
    """
    Convert ships, cargo, and shipyards data into 3D tensor. Can adapt to different grid
    sizes and player counts.

    Args:
        observation (dict): dictionary with observations from the Board object.
        shape (int): Board shape
        players_no (int): Number of players

    Returns:
        np.array: (12, 21, 21) shaped tensor. Descriptions:
            [0, :, :] - player #1 ships
            [1, :, :] - player #1 halite on the ships
            [2, :, :] - player #1 shipyards
            [3:6, :, :] - player #2
            [6:9, :, :] - player #3
            [9:12, :, :] - player #4
    """

    players_data = obs["players"]
    players_state = np.empty((3 * players_no, shape, shape))
    for i, player_data in enumerate(players_data):
        players_state[i * 3 : i * 3 + 2, :, :] = _get_ship_cargos_and_positions(
            player_data, shape
        )
        players_state[i * 3 + 2, :, :] = _get_shipyard_positions(player_data, shape)
    return players_state


def _get_shape(obs: dict) -> int:
    """
    Get the shape of a board.

    Args:
        observation (dict): dictionary with observations from the Board object.

    Returns:
        int: shape
    """
    shape = len(obs["halite"]) ** 0.5
    assert shape == int(shape)
    return int(shape)


def _get_player_no(obs: dict) -> int:
    """
    Get the number of players.

    Args:
        observation (dict): dictionary with observations from the Board object.

    Returns:
        int: players number
    """
    players = obs["players"]
    return len(players)


def _get_ship_cargos_and_positions(player_data: list, shape: int) -> np.array:
    """
    Convert ship cargo and positions into 3D tensor

    Args:
        player_data (list): list with player halite on board, shps, and shipyards
        shape (int): Board shape

    Returns:
        np.array: (2, 21, 21) shaped tensor with boolean information about the ships
            positons and the float cargo load.
    """
    ships = player_data[2]
    ships_tensor = np.zeros(shape=(2, shape, shape))
    for position_value, cargo in ships.values():
        x, y = int2pos(position_value, shape)
        ships_tensor[0, y, x] = 1.0
        ships_tensor[1, y, x] = cargo

    return ships_tensor


def _get_shipyard_positions(player_data: list, shape: int) -> np.array:
    """
    Convert shipyard positions into 2D tensor

    Args:
        player_data (list): list with player halite on board, shps, and shipyards
        shape (int): Board shape

    Returns:
        np.array: (21, 21) shaped tensor with boolean information about the shipyards
            positons.
    """
    shipyards = player_data[1]
    shipyards_tensor = np.zeros(shape=(shape, shape))
    for position_value in shipyards.values():
        x, y = int2pos(position_value, shape)
        shipyards_tensor[y, x] = 1.0

    return shipyards_tensor


def int2pos(value: int, shape: int) -> Tuple[int, int]:
    """
    Convert int value to x, y position.

    Args:
        value (int): Numerical postion interpertation
        shape (int): Board shape

    Returns:
        Tuple[int, int]: x, y coordinates
    """

    x = value % shape
    y = value // shape
    return x, y
