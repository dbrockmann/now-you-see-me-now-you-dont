
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier


def build_model(meta, hidden_layer_sizes=[256, 128]):
    """
    Builds a DNN model with fully connected layers and the ReLU activation function.

    Args:
        meta: Object containing meta information.
        hidden_layer_sizes: A list of integers, each representing the number of units in one layer.

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
            *[
                Dense(
                    units = units, 
                    activation = 'relu'
                ) for units in hidden_layer_sizes
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
