import random


def get_random_int(min_value, max_value):
    """

    :param min_value:
    :type min_value: int
    :param max_value:
    :type max_value: int
    :return:
    :rtype:
    """
    return random.randint(min_value, max_value + 1)


def get_random_position():
    return get_random_int(0, 23)
