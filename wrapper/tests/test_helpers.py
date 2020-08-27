# pylint: disable=import-error
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name

import pytest
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import (
    Board,
    Ship,
    Shipyard,
    ShipAction,
    ShipyardAction,
)

from wrapper import helpers


@pytest.fixture
def env():
    environment = make("halite", {"randomSeed": 2, "size": 5})
    agent_count = 4
    environment.reset(agent_count)
    return environment


@pytest.fixture()
def board(env):
    state = env.state
    board = Board(state[0].observation, env.configuration)
    board.ships["0-1"].next_action = ShipAction.CONVERT
    board = board.next()  # pylint: disable=not-callable
    board.shipyards["1-1"].next_action = ShipyardAction.SPAWN
    board = board.next()
    board.ships["2-1"].next_action = ShipAction.NORTH
    board = board.next()
    board.ships["2-1"].next_action = None
    board = board.next()
    board.shipyards["1-1"].next_action = ShipyardAction.SPAWN
    board = board.next()

    return board


def test_get_ship_list(board):
    result = helpers.get_ship_list(board, 0)
    assert len(result) == 2
    assert isinstance(result[0], Ship)


def test_get_shipyard_list(board):
    result = helpers.get_shipyard_list(board, 0)
    assert len(result) == 1
    assert isinstance(result[0], Shipyard)
