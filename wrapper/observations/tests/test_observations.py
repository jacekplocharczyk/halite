# pylint: disable=import-error
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name

import numpy as np
import pytest
from kaggle_environments import make

from wrapper.observations import observations


@pytest.fixture
def env():
    environment = make("halite", {"randomSeed": 2, "size": 5})
    agent_count = 4
    environment.reset(agent_count)
    return environment


def test_obs2tensor(env):

    assert False


def test__get_halite(env):
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
    obs = env.state[0]
    result = observations._get_halite(obs)
    np.testing.assert_array_equal(expected_result, result)


def test__get_shape(env):
    expected_result = 5
    obs = env.state[0]
    result = observations._get_shape(obs)
    assert expected_result == result


def test__get_player_no(env):
    expected_result = 4
    state = env.state
    result = observations._get_player_no(state)
    assert expected_result == result
