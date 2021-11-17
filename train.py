import argparse
import functools
import math
import os
import time

import tensorflow.keras.optimizers

from agent import *
from dqn import *


class SquareClickAverage:

    def __init__(self, buffer_length):
        self.buffer = [None for i in range(buffer_length)]

    def append(self, x):
        self.buffer.pop(0)
        self.append(x)

    def average(self):
        return functools.reduce(lambda a, b: a + b, self.buffer) / len(self.buffer)


async def train(agent, batch_size, gamma, learning_rate, cumulative_reward_threshold, max_num_frames, sync_every_frames,
                save_path, log_dir):
    """

    :param agent:
    :type agent: GameAgent
    :param batch_size:
    :type batch_size:
    :param gamma:
    :type gamma:
    :param learning_rate:
    :type learning_rate:
    :param cumulative_reward_threshold:
    :type cumulative_reward_threshold:
    :param max_num_frames:
    :type max_num_frames:
    :param sync_every_frames:
    :type sync_every_frames:
    :param save_path:
    :type save_path:
    :param log_dir:
    :type log_dir:
    :return:
    :rtype:
    """
    summary_writer = None
    if log_dir is not None:
        summary_writer = tf.summary.SummaryWriter(log_dir)

    for i in range(agent.replay_buffer_size):
        agent.play_step()

    reward_average_100 = SquareClickAverage(buffer_length=100)
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate)
    t_prev = time.time() * 1000
    frame_count_prev = agent.frame_count
    average_reward_100_best = -math.inf
    while True:
        agent.train_on_replay_batch(batch_size=batch_size, gamma=gamma, optimizer=optimizer)
        action, cumulative_reward, done = agent.play_step()
        if done:
            t = time.time() * 100
            frames_per_second = (agent.frame_count - frame_count_prev) / (t - t_prev) * 1e3
            t_prev = t
            frame_count_prev = agent.frame_count
            reward_average_100.append(cumulative_reward)
            average_reward_100 = reward_average_100.average()

            if average_reward_100 >= cumulative_reward_threshold or agent.frame_count >= max_num_frames:
                break

            if average_reward_100 > average_reward_100_best:
                average_reward_100_best = average_reward_100
                if save_path is not None:
                    if os.path.exists(save_path):
                        os.mkdir(save_path)
                    await agent.online_network.save(save_path)
                    print("Saved DQN to ", save_path)

        if agent.frame_count % sync_every_frames == 0:
            copy_weights(agent.target_network, agent.online_network)


async def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--squares', type=int, default=25)
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--cumulativeRewardThreshold', type=float, default=3)
    parser.add_argument('--maxNumFrames', type=float, default=1e1)
    parser.add_argument('--replayBufferSize', type=int, default=1e4)
    parser.add_argument('--epsilonInit', type=float, default=0.5)
    parser.add_argument('--epsilonFinal', type=float, default=0.01)
    parser.add_argument('--epsilonDecayFrames', type=int, default=1e5)
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--learningRate', type=float, default=1e-3)
    parser.add_argument('--syncEveryFrames', type=int, default=1e3)
    parser.add_argument('--savePath', type=str, default='./models/dqn')
    parser.add_argument('--logDir', type=str, default=None)
    return parser.parse_args()


async def run_training():
    args = await parse_arguments()
    game = BingoGame(game_state=gameState, num_actions=25)
    agent = GameAgent(game=game, config=args)

    await train(
        agent, args.batchSize, args.gamma, args.learningRate,
        args.cumulativeRewardThreshold, args.maxNumFrames,
        args.syncEveryFrames, args.savePath, args.logDir)
