
class BinaryDataset:
    """
    Binarize a dataset to Attack/Benign
    """

    def __init__(self, dataset_name):

        self.dataset_name = dataset_name

    def load(self, train, test, logger):

        train = self.binarize(train)
        test = self.binarize(test)

        logger.info('Successfully binarized %s.', self.dataset_name)

        return train, test

    def binarize(self, df):

        df = df.copy()

        df['y'] = df['y'].cat.add_categories('Attack')
        df.loc[~(df['y'] == 'Benign'), 'y'] = 'Attack'
        df['y'] = df['y'].cat.remove_unused_categories()

        return df
