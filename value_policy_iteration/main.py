#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import gym
import deeprl_hw1.lake_envs as lake_env
import numpy as np
from gym.envs.toy_text.frozen_lake import LEFT, RIGHT, DOWN, UP
import time
from deeprl_hw1 import rl  # TODO: Change this later
import matplotlib.pyplot as plt


def run_random_policy(env):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    initial_state = env.reset()
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(
            env.action_space.sample())
        env.render()

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        time.sleep(1)

    return total_reward, num_steps


def run_policy(env, policy):
    nextstate = initial_state = env.reset()
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0

    while True:
        nextstate, reward, is_terminal, debug_info = env.step(
            policy[nextstate])
        env.render()

        total_reward += (0.9 ** num_steps) * reward
        num_steps += 1

        if is_terminal:
            break

        time.sleep(0.1)

    return total_reward, num_steps


def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nS, env.nA))


def print_model_info(env, state, action):
    transition_table_row = env.P[state][action]
    print(
        ('According to transition function, '
         'taking action %s(%d) in state %d leads to'
         ' %d possible outcomes') % (lake_env.action_names[action],
                                     action, state, len(transition_table_row)))
    for prob, nextstate, reward, is_terminal in transition_table_row:
        state_type = 'terminal' if is_terminal else 'non-terminal'
        print(
            '\tTransitioning to %s state %d with probability %f and reward %f'
            % (state_type, nextstate, prob, reward))


def color_plot(value_func, m, n, fn):
    colors = plt.imshow(value_func.reshape((m, n)), cmap='viridis', interpolation='nearest')
    plt.colorbar(colors)
    plt.savefig(fn + ".jpg")


def print_for_latex(policy, m, n):
    policy_chars = np.array(list(map(lambda x: lake_env.action_names[x][0], policy))).reshape(m, n)
    for row in range(n):
        print("".join(policy_chars[row, :]))  # +r"\\")


def _main():
    # create the environment
    env = gym.make('Deterministic-4x4-FrozenLake-v0')

    print_env_info(env)
    print_model_info(env, 0, lake_env.DOWN)
    print_model_info(env, 1, lake_env.DOWN)
    print_model_info(env, 14, lake_env.RIGHT)

    input('Hit enter to run a random policy...')

    total_reward, num_steps = run_random_policy(env)
    print('Agent received total reward of: %f' % total_reward)
    print('Agent took %d steps' % num_steps)


def get_env(m, n, environment="Deterministic"):
    env_name = '%s-%sx%s-FrozenLake-v0' % (environment, m, n)
    env = gym.make(env_name)
    return env


def main(algo, m, n, environment="Deterministic"):
    import time
    # create the environment
    env_name = '%s-%sx%s-FrozenLake-v0' % (environment, m, n)
    env = gym.make(env_name)

    start_time = time.time()
    res = algo(env, 0.9)
    end_time = time.time()

    print("time to run:", (end_time - start_time) * 1000, "ms")
    print("time to run:", (end_time - start_time), "s")
    if "policy_iteration" in algo.__name__:
        policy, value_func, policy_improvement_iter, value_iter_count = res
        print("Policy improvements:", policy_improvement_iter)
        print("Value iterations:", value_iter_count)
    else:
        value_func, value_iter_count = res
        _, policy = rl.value_function_to_policy(env, 0.9, value_func)
        print("Value iterations:", value_iter_count)
        print(value_iter_count*m*n)

    print(value_func.reshape(m, n))
    print_for_latex(policy, m, n)
    color_plot(value_func, m, n, env_name + "_" + algo.__name__)

    total_reward, num_steps = run_policy(env, policy)
    print('Agent received total reward of: %f' % total_reward)
    print('Agent took %d steps' % num_steps)

    return policy, total_reward, num_steps


def run_policy_n_times(env, policy, iterations=100):
    n_rewards = []

    for i in range(iterations):
        # print("Running simulation: %s"%i)
        reward, num_steps = run_policy(env, policy)
        n_rewards.append(reward)

    avg_reward = sum(n_rewards) / float(iterations)
    print("Average reward: %s" % avg_reward)
    return avg_reward


if __name__ == '__main__':
    # m, n = 4, 4
    m, n = 8, 8

    main(rl.policy_iteration_sync, m, n, environment="Deterministic")
    # main(rl.value_iteration_sync, m, n, environment="Deterministic")

    #################### Async ############################
    # main(rl.policy_iteration_async_ordered, m, n, environment="Deterministic")
    # main(rl.policy_iteration_async_randperm, m, n, environment="Deterministic")

    # main(rl.value_iteration_async_ordered, m, n, environment="Deterministic")
    # main(rl.value_iteration_async_randperm, m, n, environment="Deterministic")

    #################### Stochastic #######################
    env = get_env(m, n, environment="Stochastic")
    # policy, _, _ = main(rl.value_iteration_sync, m, n, environment="Stochastic", )
    # policy, _, _ = main(rl.value_iteration_async_ordered, m, n, environment="Stochastic")
    # policy, _, _ = main(rl.value_iteration_async_randperm, m, n, environment="Stochastic")
    # run_policy_n_times(env, policy, 100)
    # policy, _, _ = main(rl.policy_iteration_sync, m, n, environment="Stochastic")

    #################### Custom ############################
    # policy, _, _ = main(rl.value_iteration_async_custom, m, n, environment="Deterministic")
    # policy, _, _ = main(rl.value_iteration_async_custom, m, n, environment="Stochastic")

    #################### Experiment ##########################
    # policy, _, _ = main(rl.value_iteration_async_custom, m, n, environment="Stochastic")
    # run_policy_n_times(env, policy)
    # policy, _, _ = main(rl.value_iteration_async_ordered, m, n, environment="Stochastic")
    # run_policy_n_times(env, policy)
    # policy, _, _ = main(rl.value_iteration_async_randperm, m, n, environment="Stochastic")
    # run_policy_n_times(env, policy)
