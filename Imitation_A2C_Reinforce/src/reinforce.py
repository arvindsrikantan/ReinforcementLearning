import argparse
import os
import sys

import gym
import keras
import matplotlib
import numpy as np
import time
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils
import keras.backend as K
import random

matplotlib.use('Agg')
import seaborn
# seaborn.set()
import matplotlib.pyplot as plt


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr):
        self.model = model
        self.optimizer = Adam(lr=lr)

        def custom_loss(y_true, y_pred):
            return -K.log(K.sum(y_pred * y_true, axis=1))

        self.loss_func = custom_loss
        self.model.compile(loss=self.loss_func, optimizer=self.optimizer, metrics=['accuracy'])

    def train(self, env, episodes, gamma=1.0, render=False):
        # Trains the model on a single episode using REINFORCE.
        checkpointing = 500
        test_rewards = []
        train_rewards = []
        power_gamma = {k: gamma**k for k in range(10000)}
        for episode in range(episodes + 1):
            if episode % checkpointing == 0:
                # Checkpoint
                self.save_weights("pickles/reinforce/checkpoint/iter_%s.h5" % episode)
                test_reward = []
                for _ in range(100):
                    _, _, rewards = self.generate_episode(env)
                    test_reward += [sum(rewards) * 100]
                test_rewards.append((np.array(test_reward).mean(), np.array(test_reward).std()))
                print("Average test rewards = %s" % (str(test_rewards[-1])))
                np.save("pickles/reinforce/test-rewards/iter_%s.npy" % (episode), np.array(test_rewards))
            states, actions, rewards = self.generate_episode(env, render=render)

            T = len(rewards)
            g = np.zeros(T)
            for t in reversed(range(T)):
                g[t] = sum([power_gamma[k] * rewards[k] for k in range(t, T)])

            y = np.array(np_utils.to_categorical(actions, num_classes=env.action_space.n))
            history = self.model.fit(np.array(states), y, epochs=1, batch_size=len(states), verbose=False, sample_weight=g)

            print("Episode %6d's, Steps = %3d, loss = %+.5f, cumulative reward:%+5.5f" % (episode, len(states), history.history['loss'][0], sum(rewards) * 100))
            train_rewards.append(sum(rewards) * 100)
            np.save("pickles/reinforce/train-rewards.npy", np.array(train_rewards))

    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        states = []
        actions = []
        rewards = []

        terminal = False
        state = env.reset()
        action_space = np.array(range(env.action_space.n))
        while not terminal:
            states.append(state)
            if render:
                env.render()
            # action = np.argmax(self.model.predict(np.array([state])))
            action = np.random.choice(action_space, 1, p=self.model.predict(np.array([state])).flatten())[0]
            # action = env.action_space.sample()
            state, reward, terminal, _ = env.step(action)

            actions.append(action)
            rewards.append(reward / 100.0)

        return states, actions, rewards

    def save_weights(self, name):
        self.model.save_weights(name)

    def load_weights(self, name):
        self.model.load_weights(name)
        self.model.compile(loss=categorical_crossentropy, optimizer=self.optimizer, metrics=['accuracy'])


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=100000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def createDirectories(l):
    for l in l:
        if not os.path.exists(l):
            os.makedirs(l)


def get_test_rewards(env, model):
    checkpointing = 500

    for episode in range(0, 100000, checkpointing):
        print("Episode: %s" % episode)
        model.load_weights("pickles/reinforce/checkpoint/iter_%s.h5" % (episode))
        test_rewards = []
        for _ in range(100):
            _, _, rewards = model.generate_episode(env)
            test_rewards.append(np.array(rewards))
        np.save("pickles/reinforce/test-rewards/iter_%s.npy" % episode, np.array(test_rewards))


def plot(iteration):
    r = np.load("./pickles/reinforce/test-rewards/iter_%s.npy"%(iteration))
    y, err = list(zip(*[r[i] for i in range(len(r)) if i%1==0]))
    x = list(range(0, len(y)*500, 500))
    plt.figure()
    plt.errorbar(x, y, yerr=err)
    plt.xlabel("Training episodes")
    plt.ylabel("Average reward over 100 episodes")
    plt.title("REINFORCE cumulative rewards averaged over 100 test episodes")
    plt.savefig("reinforce.png")


def test_reward_with_error():
    checkpointing = 500
    test_rewards_mean = []
    test_rewards_sd = []

    for episode in range(0, 60000, checkpointing):
        test_rewards = np.array([_temp.sum()*100 for _temp in np.load("pickles/reinforce/test-rewards/iter_%s.npy" % (episode))])
        test_rewards_mean.append(test_rewards.mean())
        test_rewards_sd.append(test_rewards.std())

    return test_rewards_mean, test_rewards_sd


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    dirs = [
        "pickles/reinforce/weights",
        "pickles/reinforce/checkpoint/",
        "pickles/reinforce/test-rewards-lists/",
        "pickles/reinforce/test-rewards/"
    ]
    createDirectories(dirs)
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    rl_model = Reinforce(model, lr)
    rl_model.train(env, num_episodes, gamma=1, render=False)
    rl_model.save_weights("./pickles/reinforce/weights/reinforce.h5")
    rl_model.load_weights("./pickles/reinforce/weights/reinforce.h5")

    plot(num_episodes)


if __name__ == '__main__':
    main(sys.argv)
