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
    observation = env.state[0]
    result = observations._get_halite(observation)  # pylint: disable=protected-access
    np.testing.assert_array_equal(expected_result, result)
