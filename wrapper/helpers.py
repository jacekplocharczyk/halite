from typing import List

from kaggle_environments.envs.halite.helpers import (
    Board,
    Ship,
    ShipAction,
    Shipyard,
    ShipyardAction,
)


def get_ship_list(board: Board, player_id: int) -> List[Ship]:
    """
    Get all ship objects of the given player.

    Args:
        board (Board): environment
        player_id (int): player id

    Returns:
        list: list with Ship objects
    """
    return _get_player_items(board.ships, player_id)


def get_shipyard_list(board: Board, player_id: int) -> List[Shipyard]:
    """
    Get all shipyard objects of the given player.

    Args:
        board (Board): environment
        player_id (int): player id

    Returns:
        list: list with Shipyard objects
    """
    return _get_player_items(board.shipyards, player_id)


def _get_player_items(items: dict, player_id: int) -> list:
    player_items = []
    for value in items.values():
        if value.player_id == player_id:
            player_items.append(value)
    return player_items


def ship_actions() -> list:
    actions = list(ShipAction)
    actions.append(None)
    return actions


def shipyard_actions() -> list:
    actions = list(ShipyardAction)
    actions.append(None)
    return actions
