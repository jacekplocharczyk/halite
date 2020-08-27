# pylint: disable=import-error
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name

import numpy as np
import pytest
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction

from wrapper.observations import observations


@pytest.fixture
def env():
    environment = make("halite", {"randomSeed": 2, "size": 5})
    agent_count = 4
    environment.reset(agent_count)
    return environment


@pytest.fixture()
def initial_board(env):
    state = env.state
    board = Board(state[0].observation, env.configuration)
    return board


@pytest.fixture()
def board(env):
    state = env.state
    board = Board(state[0].observation, env.configuration)
    board.ships["0-1"].next_action = ShipAction.CONVERT
    board = board.next()
    board.shipyards["1-1"].next_action = ShipyardAction.SPAWN
    board = board.next()
    board.ships["2-1"].next_action = ShipAction.NORTH
    board = board.next()
    board.ships["2-1"].next_action = None
    board = board.next()
    return board


def test_obs2tensor(board):

    assert False


def test__get_halite(initial_board):
    # due to a bug in the env for smaller grids cell can have more than 500 halite
    expected_result = (
        np.array(
            [
                [0, 3614, 228, 3614, 0],
                [0, 626, 0, 626, 0],
                [1490, 33, 6, 33, 1490],
                [0, 626, 0, 626, 0],
                [0, 3614, 228, 3614, 0],
            ]
        )
        / 500
    )
    obs = initial_board.observation
    result = observations._get_halite(obs)
    np.testing.assert_array_equal(expected_result, result)


def test__get_players_state(board):
    expected_result = np.array(
        [
            [
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 125, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=float,
    )
    obs = board.observation

    result = observations._get_players_state(obs)
    # checking only the first player
    np.testing.assert_array_equal(expected_result, result[:3])


def test__get_shape(board):
    expected_result = 5
    obs = board.observation
    result = observations._get_shape(obs)
    assert expected_result == result


def test__get_player_no(board):
    expected_result = 4
    obs = board.observation
    result = observations._get_player_no(obs)
    assert expected_result == result


def test__get_ship_cargos_and_positions(board):
    expected_result = np.array(
        [
            [
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 125, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=float,
    )
    obs = board.observation

    player_data = obs["players"][0]
    result = observations._get_ship_cargos_and_positions(player_data, 5)
    np.testing.assert_array_equal(expected_result, result)


def test__get_shipyard_positions(board):
    expected_result = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    obs = board.observation

    player_data = obs["players"][0]
    result = observations._get_shipyard_positions(player_data, 5)
    np.testing.assert_array_equal(expected_result, result)


@pytest.mark.parametrize(
    "position_value, shape, expected_result",
    [(1, 5, (1, 0)), (6, 5, (1, 1)), (6, 10, (6, 0))],
)
def test_int2pos(position_value, shape, expected_result):
    result = observations.int2pos(position_value, shape)
    assert result[0] == expected_result[0]
    assert result[1] == expected_result[1]

