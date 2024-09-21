
from parameters import datasets
from datasets import BinaryDataset
from utils.io import get_logger, load_dataset, save_dataset


def data_load_binary(data_folder):
    """
    Loads the datasets defined by the parameters, transforms them to binary labels, and finally saves the datasets.
    """

    binary_datasets = [dataset_name for dataset_name in datasets if isinstance(datasets[dataset_name], BinaryDataset)]
    get_logger('data-load-binary').info('Starting loading binary for the following datasets: %s.', ', '.join(binary_datasets))

    for dataset_name in binary_datasets:

        logger = get_logger('data-load-binary', data_folder, dataset_name)

        # load data
        train = load_dataset(data_folder, datasets[dataset_name].dataset_name, 'train', logger=logger)
        test = load_dataset(data_folder, datasets[dataset_name].dataset_name, 'test', logger=logger)

        # binarize data
        train, test = datasets[dataset_name].load(train, test, logger=logger)

        # save data
        save_dataset(train, data_folder, dataset_name, 'train', logger=logger)
        save_dataset(test, data_folder, dataset_name, 'test', logger=logger)
