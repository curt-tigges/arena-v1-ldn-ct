import numpy as np
import torch as t
from typing import Optional, Union
import gym
import gym.spaces
import gym.envs.registration
from gym.utils import seeding
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from utils_tabular import make_env, set_seed
from tabular_solutions import Norvig, policy_eval_exact
import solutions
from typing import Tuple
from dataclasses import asdict

def test_linear_schedule(linear_schedule):
    expected = t.tensor([solutions.linear_schedule(step, start_e=1.0, end_e=0.05, exploration_fraction=0.5, total_timesteps=500)
        for step in range(500)])
    actual = t.tensor([linear_schedule(step, start_e=1.0, end_e=0.05, exploration_fraction=0.5, total_timesteps=500) 
        for step in range(500)])
    assert expected.shape == actual.shape
    t.testing.assert_close(expected, actual)

def _random_experience(num_actions, observation_shape, num_environments):
    obs = np.random.randn(num_environments, *observation_shape)
    actions = np.random.randint(0, num_actions - 1, (num_environments,))
    rewards = np.random.randn(num_environments)
    dones = np.random.randint(0, 1, (num_environments,)).astype(bool)
    next_obs = np.random.randn(num_environments, *observation_shape)
    return (obs, actions, rewards, dones, next_obs)

def test_replay_buffer_single(
    cls, buffer_size=5, num_actions=2, observation_shape=(4,), num_environments=1, seed=1, device=t.device("cpu")
):
    """If the buffer has a single experience, that experience should always be returned when sampling."""
    rb: solutions.ReplayBuffer = cls(buffer_size, num_actions, observation_shape, num_environments, seed)
    exp = _random_experience(num_actions, observation_shape, num_environments)
    rb.add(*exp)
    for _ in range(10):
        actual = rb.sample(1, device)
        t.testing.assert_close(actual.observations, t.tensor(exp[0]))
        t.testing.assert_close(actual.actions, t.tensor(exp[1]))
        t.testing.assert_close(actual.rewards, t.tensor(exp[2]))
        t.testing.assert_close(actual.dones, t.tensor(exp[3]))
        t.testing.assert_close(actual.next_observations, t.tensor(exp[4]))

def test_replay_buffer_deterministic(
    cls, buffer_size=5, num_actions=2, observation_shape=(4,), num_environments=1, device=t.device("cpu")
):
    """The samples chosen should be deterministic, controlled by the given seed."""
    for seed in [67, 88]:
        rb: solutions.ReplayBuffer = cls(buffer_size, num_actions, observation_shape, num_environments, seed)
        rb2: solutions.ReplayBuffer = cls(buffer_size, num_actions, observation_shape, num_environments, seed)
        for _ in range(5):
            exp = _random_experience(num_actions, observation_shape, num_environments)
            rb.add(*exp)
            rb2.add(*exp)

        # Sequence of samples should be identical (ensuring they use self.rng)
        for _ in range(10):
            actual = rb.sample(2, device)
            actual2 = rb2.sample(2, device)
            for v, v2 in zip(asdict(actual).values(), asdict(actual2).values()):
                t.testing.assert_close(v, v2)

def test_replay_buffer_wraparound(
    cls, buffer_size=4, num_actions=2, observation_shape=(1,), num_environments=1, seed=3, device=t.device("cpu")
):
    """When the maximum buffer size is reached, older entries should be overwritten."""
    rb: solutions.ReplayBuffer = cls(buffer_size, num_actions, observation_shape, num_environments, seed)
    for i in range(6):
        rb.add(
            np.array([[float(i)]]),
            np.array([i % 2]),
            np.array([-float(i)]),
            np.array([False]),
            np.array([[float(i) + 1]]),
        )
    # Should be [4, 5, 2, 3] in the observations buffer now
    unique_obs = rb.sample(1000, device).observations.flatten().unique()
    t.testing.assert_close(unique_obs, t.arange(2, 6, device=device).to(dtype=unique_obs.dtype))


def test_epsilon_greedy_policy(epsilon_greedy_policy):

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", 0, 0, False, "test_eps_greedy_policy") for _ in range(5)])

    num_observations = np.array(envs.single_observation_space.shape, dtype=int).prod()
    num_actions = envs.single_action_space.n
    q_network = solutions.QNetwork(num_observations, num_actions)
    obs = t.randn((envs.num_envs, *envs.single_observation_space.shape))
    greedy_action = solutions.epsilon_greedy_policy(envs, q_network, np.random.default_rng(0), obs, 0)

    def get_actions(epsilon, seed):
        set_seed(seed)
        soln_actions = solutions.epsilon_greedy_policy(
            envs, q_network, np.random.default_rng(seed), obs, epsilon
        )
        set_seed(seed)
        their_actions = epsilon_greedy_policy(envs, q_network, np.random.default_rng(seed), obs, epsilon)
        return soln_actions, their_actions

    def are_both_greedy(soln_acts, their_acts):
        return np.array_equal(soln_acts, greedy_action) and np.array_equal(their_acts, greedy_action)

    both_actions = [get_actions(0.1, seed) for seed in range(20)]
    assert all([soln_actions.shape == their_actions.shape for (soln_actions, their_actions) in both_actions])

    both_greedy = [are_both_greedy(*get_actions(0.1, seed)) for seed in range(100)]
    assert np.mean(both_greedy) >= 0.9

    both_greedy = [are_both_greedy(*get_actions(0.5, seed)) for seed in range(100)]
    assert np.mean(both_greedy) >= 0.5

    both_greedy = [are_both_greedy(*get_actions(1, seed)) for seed in range(1000)]
    assert np.mean(both_greedy) > 0 and np.mean(both_greedy) < 0.1

# %%
from distutils.util import strtobool
import argparse

arg_help_strings = {
    "exp_name": "the name of this experiment",
    "seed": "seed of the experiment",
    "torch_deterministic": "if toggled, " "`torch.backends.cudnn.deterministic=False`",
    "cuda": "if toggled, cuda will be enabled by default",
    "track": "if toggled, this experiment will be tracked with Weights and Biases",
    "wandb_project_name": "the wandb's project name",
    "wandb_entity": "the entity (team) of wandb's project",
    "capture_video": "whether to capture videos of the agent performances (check " "out `videos` folder)",
    "env_id": "the id of the environment",
    "total_timesteps": "total timesteps of the experiments",
    "learning_rate": "the learning rate of the optimizer",
    "buffer_size": "the replay memory buffer size",
    "gamma": "the discount factor gamma",
    "target_network_frequency": "the timesteps it takes to update the target " "network",
    "batch_size": "the batch size of samples from the replay memory",
    "start_e": "the starting epsilon for exploration",
    "end_e": "the ending epsilon for exploration",
    "exploration_fraction": "the fraction of `total-timesteps` it takes from " "start-e to go end-e",
    "learning_starts": "timestep to start learning",
    "train_frequency": "the frequency of training",
    "use_target_network": "If True, use a target network.",
}
toggles = ["torch_deterministic", "cuda", "track", "capture_video"]

def parse_args(arg_help_strings=arg_help_strings, toggles=toggles):
    from w4d2_chapter4_dqn.solutions import DQNArgs
    parser = argparse.ArgumentParser()
    for (name, field) in DQNArgs.__dataclass_fields__.items():
        flag = "--" + name.replace("_", "-")
        type_function = field.type if field.type != bool else lambda x: bool(strtobool(x))
        toggle_kwargs = {"nargs": "?", "const": True} if name in toggles else {}
        parser.add_argument(
            flag, type=type_function, default=field.default, help=arg_help_strings[name], **toggle_kwargs
        )
    return DQNArgs(**vars(parser.parse_args()))
