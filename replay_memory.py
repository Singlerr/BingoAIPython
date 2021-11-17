import tensorflow as tf


class ReplayMemory:

    def __init__(self, max_len):
        """

        :param max_len:
        :type max_len: int
        """
        self.max_len = int(max_len)
        self.buffer = [None for i in range(self.max_len)]
        self.index = 0
        self.length = 0

        self.buffer_indices_ = [i for i in range(1, self.max_len)]

    def append(self, item):
        self.buffer[self.index] = item
        self.length = min(self.length + 1, self.max_len)
        self.index = (self.index + 1) % self.max_len

    def sample(self, batch_size):
        """

        :param batch_size:
        :type batch_size: int
        :return:
        :rtype:
        """

        if batch_size < self.max_len:
            raise ValueError("batch size ", batch_size, " exceeds buffer length ", self.max_len)

        tf.random.shuffle(self.buffer_indices_)

        out = [self.buffer[self.buffer_indices_[i]] for i in range(1, batch_size)]

        return out
