
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier


def build_model(meta, filter_sizes=[32, 64], kernel_size=3, hidden_layer_sizes=[256, 128], dropout=0.1):
    """
    Builds a CNN model with convolutional building blocks consisting of a convolutional layer followed by max pooling. The output is flattened and processed by fully connected layers with dropout.

    Args:
        meta: Object containing meta information.
        filter_sizes: A list of integers, each representing the number of filters in one convolutional building block.
        kernel_size: The size of the kernel for the convolutional layers.
        hidden_layer_sizes: A list of integers, each representing the number of units in one layer of the final fully connected layers.
        dropout: The dropout rate in the fully connected layers.

    Returns:
        A tensorflow model.
    """

    # fix for parallel computing
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:  # noqa: E722
        pass

    model = Sequential(
        layers = [
            Input(shape=(meta['n_features_in_'],)),
            Reshape(target_shape=(-1, 1)),
            *[
                x for filters in filter_sizes for x in
                (
                    Conv1D(
                        filters = filters, 
                        kernel_size = kernel_size,
                        padding = 'same',
                        activation = 'relu'
                    ),
                    MaxPooling1D(
                        pool_size = 2
                    )
                )
            ],
            Flatten(),
            *[
                x for units in hidden_layer_sizes for x in
                (
                    Dense(
                        units = units, 
                        activation = 'relu'
                    ),
                    Dropout(
                        rate = dropout
                    )
                )
            ],
            Dense(
                units = meta['n_classes_'],
                activation = 'softmax'
            )
        ]
    )

    return model


model = KerasClassifier(
    model = build_model,
    loss = SparseCategoricalCrossentropy,
    optimizer = Adam,
    optimizer__learning_rate = 0.001,
    metrics = [SparseCategoricalAccuracy],
    epochs = 1000,
    callbacks = EarlyStopping(
        monitor = 'val_loss',
        patience = 10,
        restore_best_weights = True,
    ),
    validation_split = 0.1,
    batch_size = 32,
    verbose = 0,
)
