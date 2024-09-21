
import os
import sys
import joblib
import logging
import pandas as pd
from pathlib import Path


def configure_logging():
    """
    Sets the basic logging configuration, including the format and handlers for console and file output.
    """

    logging.basicConfig(
        handlers = [
            logging.FileHandler('log.txt', mode='a'),
            logging.StreamHandler(stream=sys.stdout)
        ],
        level = logging.INFO,
        format = '%(asctime)s %(levelname)s %(name)s : %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )

def get_logger(flow, data_folder=None, dataset_name=None, model_name=None, attack_name=None):
    """
    Configures and returns a logger that combines the names given through the flow, dataset and model. If the dataset is given, an additional log is created at the save location.

    Args:
        flow: The name of the flow.
        data_folder: The path to the saves folder.
        dataset_name: The name of the dataset.
        model_name: The name of the model.
        attack_name: The name of the attack.

    Returns:
        A logger.
    """

    # get logger with the combined name
    logger = logging.getLogger('.'.join(filter(lambda x: x is not None, [flow, dataset_name, model_name, attack_name])))
    logger.setLevel(logging.INFO-1)

    # add handler for an additional log
    if dataset_name is not None:
        if model_name is not None:
            if attack_name is not None:
                path = os.path.join(data_folder, dataset_name, model_name, attack_name, f'{dataset_name}_{model_name}_{attack_name}_{flow}_log.txt')
            else:
                path = os.path.join(data_folder, dataset_name, model_name, f'{dataset_name}_{model_name}_{flow}_log.txt')
        else:
            path = os.path.join(data_folder, dataset_name, f'{dataset_name}_{flow}_log.txt')
        
        # create directoy and handler
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, mode='w')

        # apply root format
        handler.setFormatter(logging.root.handlers[0].formatter)
        logger.addHandler(handler)

    return logger

def save_dataset(df, data_folder, dataset_name, additional_name=None, logger=None):
    """
    Saves a dataframe to a file.

    Args:
        df: A dataframe.
        data_folder: The path to the saves folder.
        dataset_name: The name of the dataset.
        additional_name: An optional additional name to append to the name (e.g., to discriminate between test and train sets).
        logger: A logger.
    """

    file_name = dataset_name if additional_name is None else f'{dataset_name}_{additional_name}'
    path = os.path.join(data_folder, dataset_name, f'{file_name}_dataset.pkl')
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    joblib.dump(df, path, compress=True)

    if logger is not None and additional_name is None:
        logger.info('Saved the dataset to %s.', path)
    elif logger is not None:
        logger.info('Saved the dataset %s to %s.', additional_name, path)

def load_dataset(data_folder, dataset_name, additional_name=None, logger=None):
    """
    Loads a dataframe from a file.

    Args:
        data_folder: The path to the saves folder.
        dataset_name: The name of the dataset.
        additional_name: An optional additional name which is appended to the name.
        logger: A logger.

    Returns:
        The loaded dataframe.
    """

    file_name = dataset_name if additional_name is None else f'{dataset_name}_{additional_name}'
    df = joblib.load(os.path.join(data_folder, dataset_name, f'{file_name}_dataset.pkl'))

    if logger is not None:
        logger.info('Successfully loaded the dataset.')

    return df

def save_parameters(parameters, data_folder, dataset_name, model_name, logger=None):
    """
    Saves the parameters of a model to a file.

    Args:
        parameters: An object representing parameters.
        data_folder: The path to the saves folder.
        dataset_name: The name of the dataset the model was trained on.
        model_name: The name of the model.
        logger: A logger.
    """

    path = os.path.join(data_folder, dataset_name, model_name, f'{dataset_name}_{model_name}_parameters.pkl')
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    joblib.dump(parameters, path, compress=True)

    with open(os.path.join(data_folder, dataset_name, model_name, f'{dataset_name}_{model_name}_parameters_table.txt'), 'w+') as f:
        with pd.option_context('display.max_rows', None, 'display.max_colwidth', None):
            print(pd.Series(parameters), file=f)

    if logger is not None:
        logger.info('Saved the model parameters to %s.', path)

def load_parameters(data_folder, dataset_name, model_name, logger=None):
    """
    Loads the parameters of a model from a file.

    Args:
        data_folder: The path to the saves folder.
        dataset_name: The name of the dataset the model was trained on.
        model_name: The name of the model.
        logger: A logger.

    Returns:
        The loaded parameters.
    """

    parameters = joblib.load(os.path.join(data_folder, dataset_name, model_name, f'{dataset_name}_{model_name}_parameters.pkl'))

    if logger is not None:
        logger.info('Successfully loaded the parameters.')

    return parameters

def save_model(model, data_folder, dataset_name, model_name, logger=None):
    """
    Saves a model to a file.

    Args:
        model: A model.
        data_folder: The path to the saves folder.
        dataset_name: The name of the dataset the model was trained on.
        model_name: The name of the model.
        logger: A logger.
    """

    path = os.path.join(data_folder, dataset_name, model_name, f'{dataset_name}_{model_name}_model.pkl')
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=True)

    if logger is not None:
        logger.info('Saved the model to %s.', path)

def load_model(data_folder, dataset_name, model_name, logger=None):
    """
    Loads a model from a file.

    Args:
        data_folder: The path to the saves folder.
        dataset_name: The name of the dataset the model was trained on.
        model_name: The name of the model.
        logger: A logger.

    Returns:
        The loaded model.
    """

    model = joblib.load(os.path.join(data_folder, dataset_name, model_name, f'{dataset_name}_{model_name}_model.pkl'))

    if logger is not None:
        logger.info('Successfully loaded the model.')

    return model

def save_samples(samples, data_folder, dataset_name, model_name, attack_name, logger=None):
    """
    Saves samples generated from an attack together with the original samples to a file.

    Args:
        samples: A tupel of original and adversarial samples.
        data_folder: The path to the saves folder.
        dataset_name: The name of the dataset the model was trained on.
        model_name: The name of the model.
        attack_name: The name of the attack.
        logger: A logger.
    """

    path = os.path.join(data_folder, dataset_name, model_name, attack_name, f'{dataset_name}_{model_name}_{attack_name}_samples.pkl')
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    joblib.dump(samples, path, compress=True)

    if logger is not None:
        logger.info('Saved the samples to %s.', path)

def load_samples(data_folder, dataset_name, model_name, attack_name, logger=None):
    """
    Loads samples from a file.

    Args:
        data_folder: The path to the saves folder.
        dataset_name: The name of the dataset the model was trained on.
        model_name: The name of the model.
        attack_name: The name of the attack.
        logger: A logger.

    Returns:
        The loaded samples as a tuple of original and adversarial samples.
    """

    samples = joblib.load(os.path.join(data_folder, dataset_name, model_name, attack_name, f'{dataset_name}_{model_name}_{attack_name}_samples.pkl'))

    if logger is not None:
        logger.info('Successfully loaded the samples.')

    return samples
