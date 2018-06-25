#!/usr/bin/env python
from __future__ import print_function
import os
import gym
import keras, tensorflow as tf, sys, copy, argparse, pickle
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Merge, merge, Convolution2D, Flatten, Activation, BatchNormalization
from keras import backend as K
import numpy as np
import pdb
import collections
import time
import os
# from PIL import ImageOps
import cv2

"""
Authors: Arvind Srikantan (asrikan1), Ashwin NareshKumar (anareshk)
"""


def Q_loss(q_target, q_cur):
    return np.square(q_target - q_cur)


class QNetwork:

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, environment_name, network_type):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.
        self.batch_size = 32
        self.optimizer = keras.optimizers.Adam(lr=0.0001)
        self.env_name = environment_name
        self.network_type = network_type
        self.env = gym.make(environment_name)

        if self.network_type == "dqn":
            inputs = Input(self.env.observation_space.shape)
            l1 = Activation('relu')(Dense(30)(inputs))
            l3 = Activation('relu')(Dense(30)(l1))
            l5 = Activation('relu')(Dense(30)(l3))
            outputs = Activation('linear')(Dense(self.env.action_space.n)(l5))
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(optimizer=self.optimizer, loss='mse')


        elif self.network_type == "lqn":
            inputs = Input(self.env.observation_space.shape)
            outputs = Dense(self.env.action_space.n, activation='linear')(inputs)
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(optimizer=self.optimizer, loss='mse')


        elif self.network_type == "dueling_dqn":
            inputs = Input(self.env.observation_space.shape)
            l1 = Dense(24, activation='relu')(inputs)
            l3 = Dense(24, activation='relu')(l1)
            l4 = Dense(24, activation='relu')(l3)
            advantage = Dense(self.env.action_space.n, activation='linear')(l4)
            value = Dense(1, activation='linear')(l4)

            def merge_final_layer(inp):
                advantage, value = inp
                avg = K.mean(advantage)
                centered_advantage = advantage - avg
                return centered_advantage + value

            q_values = merge([advantage, value], output_shape=(self.env.action_space.n,), mode=merge_final_layer)

            self.model = Model(inputs=inputs, outputs=q_values)
            self.model.compile(optimizer=self.optimizer, loss='mse')

        elif self.network_type == "conv_dqn":
            self.model = Sequential()
            self.model.add(Convolution2D(16, 8, 8, subsample=(4, 4), input_shape=(84, 84, 4), activation='relu'))
            self.model.add(Convolution2D(32, 4, 4, subsample=(2, 2), activation='relu'))
            self.model.add(Flatten())
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dense(self.env.action_space.n))
            self.model.compile(optimizer=self.optimizer, loss='mse')
        else:
            raise Exception("Unknown network architecture")

    def save_model(self, model_file):
        # Helper function to load an existing model.
        self.model.save(model_file)

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights.
        pass

    def load_model(self, model_file):
        # Helper function to load an existing model.
        self.model = keras.models.load_model(model_file)

    def load_model_weights(self, weight_file):
        # Helper funciton to load model weights.
        pass


class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.
        self.D = collections.deque([], memory_size)
        self.burn_in = burn_in

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        return (np.array(self.D)[np.random.choice(range(len(self.D)), batch_size)]).tolist()

    def append(self, transition):
        # Appends transition to the memory.
        self.D.append(transition)


class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, gamma, epsilon, network_type, render=False):
        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.qnetwork = QNetwork(environment_name, network_type=network_type)
        self.epsilon = epsilon
        self.gamma = gamma
        self.render = render
        # the replay memory size
        self.memory_size = 50000
        # the burn in size
        self.burn_in = 10000
        self.states = []

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        q_values = q_values.flatten()
        if np.random.sample() > self.epsilon:
            return np.argmax(q_values)
        return self.qnetwork.env.action_space.sample()

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        q_values = q_values.flatten()
        return np.argmax(q_values)

    def get_state(self, current_state, buffer=False):
        # Return the current state itself if the network is not convolutional
        # Else, return 4 previous states (buffer + current state)
        if "conv" not in self.qnetwork.network_type:
            return current_state
        current_state = np.expand_dims(cv2.resize(cv2.cvtColor(current_state, cv2.COLOR_RGB2GRAY), (84, 84)), axis=2)
        if len(self.states) < 4:
            if buffer:
                self.states.append(current_state)
                temp = [np.zeros_like(current_state)] * (4 - len(self.states))
                temp += self.states
            else:
                temp = [np.zeros_like(current_state)] * (4 - len(self.states) - 1)
                temp += self.states + [current_state]
            return np.concatenate(temp, axis=2)
        else:
            if buffer:
                self.states.pop(0)
                self.states.append(current_state)
                res = np.concatenate(self.states, axis=2)
            else:
                temp = self.states[1:] + [current_state]
                res = np.concatenate(temp, axis=2)
            return res

    def simulate(self, num_episodes=20, render=False):
        # Simulate n times during training
        episode = 0
        rewards = []
        dqn = DQN_Agent(self.qnetwork.env_name, self.gamma, self.epsilon, self.qnetwork.network_type, self.render)
        dqn.qnetwork.model = keras.models.clone_model(self.qnetwork.model)
        env = gym.make(dqn.qnetwork.env_name)
        if render:
            env.render()
        while episode < num_episodes:
            state = env.reset()
            episode_r = 0
            done = False
            while not done:
                cur_state = np.array([dqn.get_state(state, buffer=True)])
                q_cur = dqn.qnetwork.model.predict(cur_state)
                action = dqn.greedy_policy(q_cur)

                state, r, done, d = env.step(action)
                episode_r += r

                if render:
                    env.render()

            episode += 1
            rewards.append(episode_r)

        return sum(rewards) / num_episodes

    def video_capture(self, iteration, name):
        # Used to capture the video of an episode by stepping through the actions suggested by the network.
        env = gym.make(self.qnetwork.env_name)
        root = 'video-captures-%s' % name
        if not os.path.exists(os.path.join(root, str(iteration))):
            os.makedirs(os.path.join(root, str(iteration)))

        directory = os.path.join(root, str(iteration))
        # Record the environment
        env = gym.wrappers.Monitor(env, directory, video_callable=lambda episode_id: True)  # ,force=True
        done = False
        state = env.reset()

        while not done:
            q_cur = self.qnetwork.model.predict(np.array([self.get_state(state, buffer=True)]))
            action = self.greedy_policy(q_cur)

            state, r, done, d = env.step(action)

        return

    def train(self, max_iterations=100, with_replay=False, use_target=False):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        env = self.qnetwork.env
        max_iterations += 1
        episodes = 0
        min_q = 0

        iterations = 0
        rewards = []
        iters = []
        vid_checkpoints = [0, int(max_iterations / 3), int(2 * max_iterations / 3), max_iterations - 1]
        losses = []

        sim_count = 0
        episodes = 0

        reward_q = collections.deque([], 100)

        if not with_replay:
            # If we need not use replay memory
            # Save weights every checkpointing number of iterations
            checkpointing = 5000
            setup = "noreplay_" + self.qnetwork.network_type
            while iterations < max_iterations:
                state = env.reset()
                r = 0.0
                terminal = False
                while not terminal and iterations < max_iterations:
                    if self.render == True:
                        env.render()
                    cur_state = np.array([self.get_state(state, buffer=True)])
                    q_cur = self.qnetwork.model.predict(cur_state)
                    action = self.epsilon_greedy_policy(q_cur)
                    next_state, r, terminal, d = env.step(action)
                    q_next = self.qnetwork.model.predict(np.array([self.get_state(next_state)]))
                    target = q_cur.copy()
                    if "MountainCar-v0" in self.qnetwork.env_name and self.qnetwork.network_type == "lqn":
                        # Suggested edit from piazza for MountainCar with lqn
                        if next_state[0] >= 0.5:
                            target[0, action] = r
                        else:
                            target[0, action] = r + self.gamma * q_next.max()
                    else:
                        if not terminal:
                            target[0, action] = r + self.gamma * q_next.max()
                        else:
                            target[0, action] = r

                    history = self.qnetwork.model.fit(cur_state, target, verbose=False, epochs=1)

                    state = next_state
                    if iterations > checkpointing and iterations % checkpointing == 0:
                        # Checkpointing
                        self.qnetwork.save_model(
                            "./model-checkpoints_%s_%s/%s" % (self.qnetwork.env_name, setup, iterations))
                        np.save("./model-loss_%s_%s/%s" % (self.qnetwork.env_name, setup, iterations), np.array(losses))

                        # Simulating
                        print('Sim count : ', sim_count, 'iter count : ', iterations)
                        rewards.append(self.simulate(num_episodes=20))
                        iters.append(iterations)
                        np.save("./model-rewards_%s_%s/%s" % (self.qnetwork.env_name, setup, iterations),
                                np.array(rewards))
                        print(rewards)

                        # Adding loss
                        loss = history.history["loss"]
                        losses.append(loss)
                        sim_count += 1

                    # if iterations in vid_checkpoints:
                    #     self.video_capture(iterations)

                    iterations += 1
                    if self.epsilon > 0.05:
                        self.epsilon -= (0.5 - 0.05) / (1e6)

                episodes += 1

                print("episode=%s, Iteration = %s, loss = %s, epsilon = %s" % (
                    episodes, iterations, history.history["loss"], self.epsilon))

        else:
            checkpointing = 1000
            replay_memory = self.burn_in_memory()
            target_network = None
            episodes = 0
            update_stage = 10000
            setup = "replay_" + self.qnetwork.network_type
            c = 0
            while c < max_iterations:
                r_sum = 0
                state = env.reset()
                terminal = False
                time = 0

                while not terminal:
                    if c % checkpointing == 0:
                        # Checkpointing
                        self.qnetwork.save_model("./model-checkpoints_%s_%s/%s" % (self.qnetwork.env_name, setup, c))
                        np.save("./model-loss_%s_%s/%s" % (self.qnetwork.env_name, setup, c), np.array(losses))

                        # Simulating
                        print('Sim count : ', sim_count, 'iter count : ', c)
                        rewards.append(self.simulate(num_episodes=20))
                        print(rewards)
                        iters.append(iterations)
                        np.save("./model-rewards_%s_%s/%s" % (self.qnetwork.env_name, setup, c), np.array(rewards))

                        # Adding loss
                        if c > 0:
                            loss = history.history["loss"]
                            losses.append(loss)
                        sim_count += 1

                    if self.render == True:
                        env.render()
                    if c % update_stage == 0:
                        if use_target:
                            target_network = keras.models.clone_model(self.qnetwork.model)
                        else:
                            target_network = self.qnetwork.model
                    c += 1
                    time += 1

                    cur_state = self.get_state(state, buffer=True)
                    q_vals = self.qnetwork.model.predict(np.array([cur_state]))
                    min_q = min(q_vals.min(), min_q)
                    action = self.epsilon_greedy_policy(q_vals)
                    next_state, r, terminal, _ = env.step(action)
                    replay_memory.append((cur_state, action, r, self.get_state(next_state), terminal))

                    # Get the targets for the batch
                    samples = replay_memory.sample_batch(self.qnetwork.batch_size)
                    X = []
                    y = []
                    for cur_s, act, _r, next_s, term in samples:
                        X.append(cur_s)
                        if term:
                            q_cur = self.qnetwork.model.predict(np.array([cur_s]))
                            target = q_cur.flatten()
                            target[act] = _r
                            y.append(target)
                        else:
                            q_cur = self.qnetwork.model.predict(np.array([cur_s]))
                            q_next_target = target_network.predict(np.array([next_s]))
                            q_next_current = self.qnetwork.model.predict(np.array([next_s]))
                            a_max = np.argmax(q_next_current.flatten())
                            target = q_cur.flatten()
                            target[act] = _r + self.gamma * q_next_target.flatten()[a_max]
                            y.append(target)
                    # train on the batch
                    history = self.qnetwork.model.fit(np.array(X), np.array(y), verbose=False)

                    state = next_state

                    if self.epsilon > 0.05:
                        self.epsilon -= (0.5 - 0.05) / (1e6)
                    r_sum += r

                episodes += 1
                reward_q.append(r_sum)

                print("episode=%s, Iteration = %s, loss = %s, epsilon = %s, avg_reward = %s" % (
                    episodes, c, history.history["loss"], self.epsilon, sum(reward_q) / float(len(reward_q))))

    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cumulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        env = gym.make(self.qnetwork.env_name)

        rewards = []
        for episode in range(100):
            state = env.reset()
            terminal = False
            rewards.append(0.0)

            if self.render:
                time.sleep(1)
                env.render()
                time.sleep(0.1)
            iterations = 0
            while not terminal:
                q_cur = self.qnetwork.model.predict(np.array([self.get_state(state, buffer=True)]))
                action = self.greedy_policy(q_cur)
                state, r, terminal, d = env.step(action)
                rewards[-1] += r

                if self.render:
                    env.render()
                iterations += 1
            print("iterations=%s" % iterations)

        print("Average reward = %s" % (sum(rewards) / float(len(rewards))))
        print("Standard deviation = %s" % np.std(np.array(rewards)))

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        replay_memory = Replay_Memory(self.memory_size, self.burn_in)
        terminal = True
        env = self.qnetwork.env
        state = None
        for _ in range(self.burn_in):
            if terminal:
                state = env.reset()
                terminal = False
            cur_state = self.get_state(state, buffer=True)
            q_vals = self.qnetwork.model.predict(np.array([cur_state]))
            action = self.epsilon_greedy_policy(q_vals)

            next_state, r, terminal, _ = env.step(action)

            replay_memory.append((cur_state, action, r, self.get_state(next_state), terminal))
            state = next_state

        return replay_memory


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    parser.add_argument('--network', dest='network', type=str)
    parser.add_argument('--replay', dest='replay', type=int, default=1)
    parser.add_argument('--ddqn', dest='with_target', type=int, default=0)
    return parser.parse_args()


def main(args):
    import os

    args = parse_arguments()
    environment_name = args.env
    gamma = 1 if environment_name == "MountainCar-v0" else 0.99
    network_type = args.network  # "lqn"
    with_replay = False if args.replay == 0 else True
    rep = 'replay_' if with_replay else 'noreplay_'
    setup = rep + network_type

    if not os.path.exists("./models"):
        os.makedirs("./models")

    if not os.path.exists("./model-checkpoints_%s_%s" % (environment_name, setup)):
        os.makedirs("./model-checkpoints_%s_%s" % (environment_name, setup))

    if not os.path.exists("./model-loss_%s_%s" % (environment_name, setup)):
        os.makedirs("./model-loss_%s_%s" % (environment_name, setup))

    if not os.path.exists("./model-rewards_%s_%s" % (environment_name, setup)):
        os.makedirs("./model-rewards_%s_%s" % (environment_name, setup))

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)

    agent = DQN_Agent(environment_name, gamma, 0.5, network_type, render=(False if args.render == 0 else True))
    if args.train == 1:
        agent.train(max_iterations=1000000, with_replay=with_replay,
                    use_target=(False if args.with_target == 0 else True))
        agent.qnetwork.save_model(args.model_file)
    else:
        agent.qnetwork.load_model(args.model_file)
    agent.test()


if __name__ == '__main__':
    main(sys.argv)
