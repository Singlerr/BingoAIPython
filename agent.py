import random

import tensorflow

import replay_memory
import utils

import game_graphic

from game import *


class GameAgent:

    def __init__(self, game, config):
        """

        :param game:
        :type game: BingoGame
        :param config:
        :type config: argparse.ArgumentParser
        """
        self.cumulative_reward_ = 0
        self.game = game

        self.epsilon_init = config.epsilonInit
        self.epsilon_final = config.epsilonFinal
        self.epsilon_decay_frames = config.epsilonDecayFrames
        self.epsilon_increment_ = (self.epsilon_final - self.epsilon_init) / self.epsilon_decay_frames

        self.online_network = dqn.create_dqn(SQUARE_SIZE)
        self.target_network = dqn.create_dqn(SQUARE_SIZE)
        self.optimizer = tf.optimizers.Adam(config.learningRate)

        self.replay_buffer_size = int(config.replayBufferSize)
        self.replay_memory = replay_memory.ReplayMemory(config.replayBufferSize)

        self.frame_count = 0

    def reset(self):
        self.cumulative_reward_ = 0
        self.game.reset()

    def play_step(self):

        self.epsilon = self.epsilon_final if self.frame_count >= self.epsilon_decay_frames else self.epsilon_init + self.epsilon_increment_ * self.frame_count

        self.frame_count += 1

        state = self.game.get_state()
        action = None

        if random.random() < self.epsilon:
            action = utils.get_random_position()
        else:
            state_tensor = get_state_tensor(state)
            action = self.online_network.predict(state_tensor).argmax(-1)[0]

        next_state, done, reward = step(action)
        print("Current reward: ", reward, ", done: ", done, ", action: ", action)

        self.replay_memory.append([state, action, reward, done, next_state])
        self.cumulative_reward_ += reward

        if done:
            self.reset()

        return action, self.cumulative_reward_, done

    def train_on_replay_batch(self, batch_size, gamma, optimizer):

        """

        :param batch_size:
        :type batch_size: int
        :param gamma:
        :type gamma:
        :param optimizer:
        :type optimizer:
        :return:
        :rtype:
        """
        batch = self.replay_memory.sample(batch_size)

        with tf.GradientTape() as tape:
            state_tensor = get_state_tensor(map(lambda example: example[0], batch))
            action_tensor = tf.Tensor(map(lambda example: example[4], batch), dtype='int32')

            qs = self.online_network.apply(state_tensor, True).mul(tf.one_hot(action_tensor, SQUARE_SIZE)).sum(-1)

            reward_tensor = tf.Tensor(map(lambda example: example[2], batch))
            next_state_tensor = get_state_tensor(map(lambda example: example[4], batch))
            next_max_q_tensor = self.target_network.predict(next_state_tensor).max(-1)

            done_mask = tf.summary.scalar(1).sub(tf.Tensor(map(lambda example: example[3], batch))).as_type('float32')
            target_qs = reward_tensor + (next_max_q_tensor * done_mask * gamma)
            loss = tensorflow.keras.losses.MeanSquaredError(target_qs, qs)
        grads = tape.gradient(loss)
        optimizer.apply_gradients(grads.grads)
        tf.dispose(grads)
