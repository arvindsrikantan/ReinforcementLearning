import argparse
import os
import sys

import gym
import keras
import matplotlib
import numpy as np
from keras.layers import Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce


class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, model, lr, critic_model, critic_lr, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.critic_model = critic_model
        self.n = n
        self.lr = lr
        super().__init__(model, lr)
        self.critic_optimizer = Adam(lr=critic_lr)
        self.critic_model.compile(loss='mean_squared_error', optimizer=self.critic_optimizer, metrics=['accuracy'])

    def train(self, env, episodes, gamma=1.0, render=False):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        checkpointing = 500
        test_rewards = []
        train_rewards = []
        power_gamma = {k: gamma ** k for k in range(10000)}
        for episode in range(episodes + 1):
            # self.model.compile(loss=categorical_crossentropy, optimizer=self.optimizer, metrics=['accuracy'])
            if episode % checkpointing == 0:
                # Checkpoint
                self.save_weights("pickles/a2c/checkpoint/%s_n_%s_iter_%s.h5" % ("%s", self.n, episode))
                test_reward = []
                for _ in range(100):
                    _, _, rewards = self.generate_episode(env)
                    test_reward += [sum(rewards) * 100]
                test_rewards.append((np.array(test_reward).mean(), np.array(test_reward).std()))
                print("Average test rewards = %s" % (str(test_rewards[-1])))
                np.save("pickles/a2c/test-rewards/n_%s_iter_%s.npy" % (self.n, episode), np.array(test_rewards))
            states, actions, rewards = self.generate_episode(env, render=render)
            
            r = np.zeros(len(rewards))
            g = np.zeros(len(rewards))
            
            v = np.zeros(len(rewards))
            T = len(rewards)
            v = self.critic_model.predict(np.array(states)).flatten()
            
            for t in reversed(range(T)):
                v_end = 0 if (t + self.n >= T) else v[t + self.n]
                r[t] = power_gamma[self.n]*v_end + sum([(power_gamma[k]*rewards[t + k] if (t + k < T) else 0) for k in range(self.n)])
                g[t] = r[t] - v[t]
            
            history = self.model.fit(np.array(states),
                                     np.array(np_utils.to_categorical(actions, num_classes=env.action_space.n)),
                                     epochs=1, batch_size=len(states), verbose=False, sample_weight=g)
            critic_history = self.critic_model.fit(np.array(states), r, epochs=1, batch_size=len(states), verbose=False)

            print("Episode %6d's, Steps = %3d, loss = %+.5f, critic_loss = %+.5f, cumulative reward:%+5.5f" % (
                episode, len(states), history.history['loss'][0], critic_history.history['loss'][0],
                sum(rewards) * 100))
            train_rewards.append(sum(rewards) * 100)
            np.save("pickles/a2c/n_%s_train-rewards.npy" % self.n, np.array(train_rewards))

    def save_weights(self, name):
        self.model.save_weights(name % "actor")
        self.critic_model.save_weights(name % "critic")

    def load_weights(self, name):
        self.model.load_weights(name % "actor")
        self.model.compile(loss=categorical_crossentropy, optimizer=self.optimizer, metrics=['accuracy'])
        self.critic_model.load_weights(name % "critic")
        self.critic_model.compile(loss='mean_squared_error', optimizer=self.critic_optimizer, metrics=['accuracy'])


def createDirectories(l):
    for l in l:
        if not os.path.exists(l):
            os.makedirs(l)


def get_test_rewards(env, model, n=1):
    checkpointing = 500
    for episode in range(0, 60000, checkpointing):
        print("Episode: %s" % episode)
        model.load_weights("pickles/a2c/checkpoint/%s_n_%s_iter_%s.h5" % ("%s", n, episode))
        test_rewards = []
        for _ in range(100):
            _, _, rewards = model.generate_episode(env)
            test_rewards.append(np.array(rewards))
        np.save("pickles/a2c/test-rewards-lists/n_%s_iter_%s.npy" % (n, episode), np.array(test_rewards))


def test_reward_with_error(n=1):
    checkpointing = 500
    test_rewards_mean = []
    test_rewards_sd = []

    for episode in range(0, 60000, checkpointing):
        test_rewards = np.array([_temp.sum()*100 for _temp in np.load("pickles/a2c/test-rewards-lists/n_%s_iter_%s.npy" % (n, episode))])
        test_rewards_mean.append(test_rewards.mean())
        test_rewards_sd.append(test_rewards.std())

    return test_rewards_mean, test_rewards_sd


def plot(n, iteration):
    # n=20
    # iteration = 86500 # n=100
    # iteration = 97000 # n = 1
    # iteration = 100000
    r = np.load("./pickles/a2c/test-rewards/n_%s_iter_%s.npy"%(n, iteration))
    y, err = list(zip(*[r[i] for i in range(len(r)) if i % 2 == 0]))
    x = list(range(0, len(y) * 1000, 1000))
    # x = list(range(0, len(y)*500, 500))
    plt.figure()
    plt.errorbar(x, y, yerr=err)
    plt.xlabel("Training episodes")
    plt.ylabel("Average reward over 100 episodes")
    plt.title("A2C cumulative reward for N=%s averaged over 100 episodes" % n)
    plt.savefig("a2c_n_%s.png" % n)


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=100000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=1e-3, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-3, help="The critic's learning rate.")# 5e-4 before
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")

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


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    dirs = [
        "pickles/a2c/weights",
        "pickles/a2c/checkpoint/",
        "pickles/a2c/test-rewards/",
        "pickles/a2c/test-rewards-lists/"
    ]
    createDirectories(dirs)
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Load the actor model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    with open(model_config_path, 'r') as f:
        critic_model = keras.models.model_from_json(f.read())
        critic_model.pop()
        critic_model.add(Dense(30, name='dense_4'))
        critic_model.add(Dense(1, name='dense_5'))


    a2c = A2C(model, lr, critic_model, critic_lr, n)
    a2c.train(env, episodes=num_episodes, render=render)

    plot(n, num_episodes)


if __name__ == '__main__':
    main(sys.argv)
