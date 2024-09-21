
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import L1
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CosineSimilarity
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, ClassifierMixin
from scikeras.wrappers import BaseWrapper, KerasClassifier


def build_ae(meta, hidden_layer_sizes=[256, 128], latent_size=0.7, l1=0.001):
    """
    Builds an AE model consisting of an encoder and a decoder. The encoder applies an activation penalty on its last layer to obtain sparse representations.

    Args:
        meta: Object containing meta information.
        hidden_layer_sizes: A list of integers, each representing the number of units in one layer of the encoder. The layer architecture is mirrored for the decoder.
        latent_size: The size of the last layer of the encoder. Either an Integer representing the number or a float from which the size is computed in relation to the input size (compression rate).
        l1: Activation penalty factor for the last layer of the encoder.

    Returns:
        A tensorflow model.
    """

    # fix for parallel computing
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:  # noqa: E722
        pass

    # compute latent size if given as the compression rate
    if isinstance(latent_size, float):
        latent_size = int(latent_size * meta['n_features_in_'])

    encoder = Sequential(
        layers = [
            Input(shape=(meta['n_features_in_'],)),
            *[
                Dense(
                    units = units, 
                    activation = 'relu',
                ) for units in hidden_layer_sizes
            ],
            Dense(
                units = latent_size,
                activation = 'relu',
                activity_regularizer = L1(
                    l1 = l1
                )
            )
        ]
    )

    decoder = Sequential(
        layers = [
            Input(shape=(latent_size,)),
            *[
                Dense(
                    units = units, 
                    activation = 'relu',
                ) for units in reversed(hidden_layer_sizes)
            ],
            Dense(
                units = meta['n_features_in_'],
                activation = 'sigmoid'
            )
        ]
    )

    model = Sequential([encoder, decoder])

    return model


def build_ae_classifier(meta, encoder, hidden_layer_sizes=[256, 128]):
    """
    Builds a fully connected classifier using a trained encoder.

    Args:
        meta: Object containing meta information.
        encoder: A trained Tensorflow model.
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

    inputs = Input(shape=(meta['n_features_in_'],))
    classifier = Sequential(
        layers = [
            Input(shape=(encoder.output_shape[-1],)),
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

    model = Sequential([inputs, encoder, classifier])

    return model


class AutoencoderClassifier(BaseEstimator, ClassifierMixin):
    """
    Combines the Autoencoder architecture with a fully connected classifier to be used in a sklearn pipeline.
    """

    def __init__(self, autoencoder, classifier, verbose=0):
        """
        Initialize with Scikeras models.

        Args:
            autoencoder: A Scikeras model.
            classifier: A Scikeras model.
            verbose: Verbosity value.
        """

        self.autoencoder = autoencoder
        self.classifier = classifier
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fits the underlying models by first fitting the autoencoder and then fitting the classifier using the fitted encoder part of the autoencoder. The encoder is freezed in the classifier.

        Args:
            X: The training data.
            y: The corresponding true labels.
        """

        # set verbosity values
        self.autoencoder.set_params(verbose=self.verbose)
        self.classifier.set_params(verbose=self.verbose)

        # train autoencoder and freeze encoder
        self.autoencoder.fit(X, X)
        encoder = self.autoencoder.model_.get_layer(index=0)
        encoder.trainable = False

        # train classifier with encoder
        self.classifier.set_params(model__encoder=encoder)
        self.classifier.fit(X, y)
        self.classifier.set_params(model__encoder=None)

        # expose attributes from classifier
        self.model_ = self.classifier.model_
        self.classes_ = self.classifier.classes_
        self.target_encoder_ = self.classifier.target_encoder_
        self.n_features_in_ = self.classifier.n_features_in_
        self.history_ = self.classifier.history_


    def predict(self, X):
        """
        Predicts a class given data by utilizing the trained classifier.

        Args:
            X: The data.

        Returns:
            The predictions.
        """
        prediction = self.classifier.predict(X)

        return prediction


model = AutoencoderClassifier(
    autoencoder = BaseWrapper(
        model = build_ae,
        loss = CosineSimilarity,
        optimizer = Adam,
        optimizer__learning_rate = 0.001,
        epochs = 1000,
        callbacks = EarlyStopping(
            monitor = 'val_loss',
            patience = 10,
            restore_best_weights = True,
        ),
        validation_split = 0.1,
        batch_size = 32,
    ), 
    classifier = KerasClassifier(
        model = build_ae_classifier,
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
    ),
    verbose = 0,
)
