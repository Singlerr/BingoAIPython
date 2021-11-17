import keras
from tensorflow.keras import layers


def create_dqn(num_actions):
    model = keras.Sequential()
    model.add(
        layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', input_shape=[25, 9, 9, 1]))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=100, activation='relu'))
    model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(units=num_actions))

    return model


def copy_weights(dest_network, src_network):
    """
    
    :param dest_network: 
    :type dest_network: keras.engine.sequential.Sequential
    :param src_network: 
    :type src_network: tensorflow.keras.Model
    :return: 
    :rtype: 
    """

    original_dest_network_trainable = None
    if dest_network.trainable is not src_network.trainable:
        original_dest_network_trainable = dest_network.trainable
        dest_network.trainable = src_network.trainable

    dest_network.set_weights(src_network.get_weights())

    if original_dest_network_trainable is not None:
        dest_network.trainable = original_dest_network_trainable
