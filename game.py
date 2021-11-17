import numpy as np
import tensorflow as tf

import dqn

global SQUARE_SIZE

SQUARE_SIZE = 25

global HORIZONTAL_BINGO

HORIZONTAL_BINGO = 9

global VERTICAL_BINGO

VERTICAL_BINGO = 9

global MAX_STEP

MAX_STEP = 20


def num_new_bingo(preIndexes, indexes):
    """

    :param preIndexes:
    :type preIndexes: list
    :param indexes:
    :type indexes: list
    :return:
    :rtype:
    """
    result = [(not preIndexes[i]) and indexes[i] for i in range(len(indexes))]
    return result.count(True)


def find_horizontal_bingo(i, squares):
    """

    :param i:
    :type i: int
    :param squares:
    :type squares: list
    :return:
    :rtype:
    """
    start = i // 5
    for k in range(start, start + 5):
        if squares[k] != 1:
            return False, -1
    return True, start


def find_vertical_bingo(i, squares):
    """

    :param i:
    :type i: int
    :param squares:
    :type squares: list
    :return:
    :rtype:
    """
    start = i % 5
    for k in range(start, 25, 5):
        if squares[k] != 1:
            return False, -1
    return True, start


def find_horizontal_total_bingo(i):
    """

    :param i:
    :type i: int
    :return:
    :rtype:
    """
    squares = list(gameState.squares)
    h_indexes = list(gameState.horizontal_bingo)

    b = False

    pos = [i, i - 1, i + 1, i - 5, i + 5]

    for p in pos:
        if p < 0 or p > 24:
            continue

        bingo, index = find_horizontal_bingo(i, squares)
        b = b or bingo
        if bingo:
            h_indexes[index] = bingo

    return b, h_indexes


def find_vertical_total_bingo(i):
    """

    :param i:
    :type i: int
    :return:
    :rtype:
    """
    squares = list(gameState.squares)
    v_indexes = list(gameState.vertical_bingo)

    b = False

    pos = [i, i - 1, i + 1, i - 5, i + 5]

    for p in pos:
        if p < 0 or p > 24:
            continue

        bingo, index = find_vertical_bingo(i, squares)
        b = b or bingo
        if bingo:
            v_indexes[index] = bingo

    return b, v_indexes


def find_bingo(i):
    """

    :param i:
    :type i: int
    :return:
    :rtype:
    """

    h_bingo, h_indexes = find_horizontal_total_bingo(i)
    v_bingo, v_indexes = find_vertical_total_bingo(i)
    return h_bingo or v_bingo, h_indexes, v_indexes


def handle_click(i, squares_origin):
    """

    :param i:
    :type i: int
    :param squares_origin:
    :type squares_origin: list
    """
    squares = list(squares_origin)

    if squares[i] == 1:
        return squares

    if (i + 1) // 5 == 0:
        down = squares[i + 5]

        if (i + 1) % 5 == 1:
            right = squares[i + 1]

            if right == 1:
                squares[i + 1] = 0
            else:
                squares[i + 1] = 1
            if down == 1:
                squares[i + 5] = 0
            else:
                squares[i + 5] = 1
        elif (i + 1) % 5 == 0:
            left = squares[i - 1]
            if left == 1:
                squares[i - 1] = 0
            else:
                squares[i - 1] = 1
            if down == 1:
                squares[i + 5] = 0
            else:
                squares[i + 5] = 1
        else:
            right = squares[i + 1]

            if right == 1:
                squares[i + 1] = 0
            else:
                squares[i + 1] = 1
            if down == 1:
                squares[i + 5] = 0
            else:
                squares[i + 5] = 1

            left = squares[i - 1]

            if left == 1:
                squares[i - 1] = 0
            else:
                squares[i - 1] = 1
            if down == 1:
                squares[i + 5] = 0
            else:
                squares[i + 5] = 1
    elif i // 5 == 4:
        up = squares[i - 5]
        if (i + 1) % 5 == 1:
            right = squares[i + 1]
            if right == 1:
                squares[i + 1] = 0
            else:
                squares[i + 1] = 1
            if up == 1:
                squares[i - 5] = 0
            else:
                squares[i - 5] = 1
        elif (i + 1) % 5 == 0:
            left = squares[i - 1]
            if left == 1:
                squares[i - 1] = 0
            else:
                squares[i - 1] = 1
            if up == 1:
                squares[i - 5] = 0
            else:
                squares[i - 5] = 1
        else:
            right = squares[i + 1]
            if right == 1:
                squares[i + 1] = 0
            else:
                squares[i + 1] = 1
            if up == 1:
                squares[i - 5] = 0
            else:
                squares[i - 5] = 1
            left = squares[i - 1]
            if left == 1:
                squares[i - 1] = 0
            else:
                squares[i - 1] = 1

            if up == 1:
                squares[i - 5] = 0
            else:
                squares[i - 5] = 0
    else:
        up = squares[i - 5]
        down = squares[i + 5]
        if (i + 1) % 5 == 1:
            right = squares[i + 1]
            if right == 1:
                squares[i + 1] = 0
            else:
                squares[i + 1] = 1
            if up == 1:
                squares[i - 5] = 0
            else:
                squares[i - 5] = 1
            if down == 1:
                squares[i + 5] = 0
            else:
                squares[i + 5] = 1
        elif (i + 1) % 5 == 0:
            left = squares[i - 1]
            if left == 1:
                squares[i - 1] = 0
            else:
                squares[i - 1] = 1
            if up == 1:
                squares[i - 5] = 0
            else:
                squares[i - 5] = 1
            if down == 1:
                squares[i + 5] = 0
            else:
                squares[i + 5] = 1
        else:
            right = squares[i + 1]
            if right == 1:
                squares[i + 1] = 0
            else:
                squares[i + 1] = 1
            if up == 1:
                squares[i - 5] = 0
            else:
                squares[i - 5] = 1

            left = squares[i - 1]
            if left == 1:
                squares[i - 1] = 0
            else:
                squares[i - 1] = 1
            if up == 1:
                squares[i - 5] = 0
            else:
                squares[i - 5] = 1
            if down == 1:
                squares[i + 5] = 0
            else:
                squares[i + 5] = 1
    squares[i] = 1
    return squares


class GameState:
    def __init__(self, squares=[0 for i in range(SQUARE_SIZE)], vertical_bingo=[False for i in range(VERTICAL_BINGO)],
                 horizontal_bingo=[False for i in range(HORIZONTAL_BINGO)], game_step=0):
        self.squares = squares
        self.vertical_bingo = vertical_bingo
        self.horizontal_bingo = horizontal_bingo
        self.step = game_step

    def reset(self):
        self.squares = [0 for i in range(SQUARE_SIZE)]
        self.vertical_bingo = [False for i in range(VERTICAL_BINGO)]
        self.horizontal_bingo = [False for i in range(HORIZONTAL_BINGO)]
        self.step = 0

    def clone(self):
        return GameState(self.squares, self.vertical_bingo, self.horizontal_bingo, self.step)


global gameState
gameState = GameState()


def step(best_loc):
    done = False
    if gameState.squares[best_loc] == 1:
        done = True
        return gameState, done, -2

    squares = handle_click(best_loc, gameState.squares)

    gameState.squares = squares

    bingo, h_indices, v_indices = find_bingo(best_loc)
    reward = num_new_bingo(gameState.horizontal_bingo, h_indices) + num_new_bingo(gameState.vertical_bingo, v_indices)

    if gameState.step % 3 == 0 and reward > 0:
        reward -= 1
        done = True

    gameState.horizontal_bingo = h_indices
    gameState.vertical_bingo = v_indices
    gameState.step = ++gameState.step

    return gameState, done, reward


def get_state_tensor(state):
    """
    :param state:
    :type state: GameState
    :return:
    :rtype: Tensor
    """
    if isinstance(state, list) is False:
        state = [state]
    num_examples = len(state)

    # (number of examples, square value, horizontal bingo, vertical bingo, step(num examples))
    buffer = np.zeros(shape=[num_examples, SQUARE_SIZE, HORIZONTAL_BINGO, VERTICAL_BINGO, num_examples])
    for n in range(num_examples):
        if state[n] is None:
            continue
        for i in range(len(state[n].squares)):
            s = state[n].squares[i]
            for j in range(HORIZONTAL_BINGO):
                buffer[n, i, j, j, n] = s + \
                                        state[n].horizontal_bingo[j] + \
                                        state[n].vertical_bingo[j] + \
                                        state[n].step
    return tf.convert_to_tensor(buffer)


class BingoGame:

    def __init__(self, game_state, num_actions):
        """
        :param game_state
        :type GameState
        """

        self.model = dqn.create_dqn(num_actions)
        self.game_state = game_state
        self.num_actions = num_actions

    def get_state(self):
        return self.game_state.clone()

    def reset(self):
        self.game_state.reset()
