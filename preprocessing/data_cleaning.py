
import numpy as np

from utils.visualization import visualize_correlation_matrix, visualize_duplicates


def clean_data(df, data_folder, dataset_name, logger):
    """
    
    """

    logger.info('Starting data cleaning.')

    # remove features without information
    df = remove_no_information_features(df, logger=logger)

    # remove redundant features
    df = remove_redundant_features(df, data_folder=data_folder, dataset_name=dataset_name, logger=logger)

    # remove rows without label
    df = remove_no_label_rows(df, logger=logger)

    # remove rows with missing values (or replace for categoric data)
    df = remove_missing_value_rows(df, logger=logger)

    # remove rows with infinite
    df = remove_infinite_value_rows(df, logger=logger)

    # remove rows with negative values
    df = remove_negative_value_rows(df, logger=logger)

    # remove too small classes for oversampling
    df = remove_too_small_classes(df, logger=logger)

    logger.info('Finished data cleaning. The resulting dataset contains %d rows with %d classes and %d features. The classes are: %s. The features are: %s.', len(df.index), len(df['y'].cat.categories), len(df.drop(columns='y').columns), ', '.join(df['y'].cat.categories), ', '.join(df.drop(columns='y').columns))

    return df

def remove_no_information_features(df, logger):
    """
    Removes constant features that include no information. For numeric features, remove with a standard deviation of 0. For categoric features, remove if only a single category is present.
    """

    # calculate standard deviations
    std = df.std(numeric_only=True)

    # find features with standard deviation of 0
    no_information_features = list(std[std == 0.0].index)

    # search categorical features for constant values
    for c in df.select_dtypes(include='category').columns:
        if c != 'y' and len(df[c].cat.categories) <= 1:
            no_information_features.append(c)

    # remove features with a standard deviation of 0 or only a single category
    filtered_df = df.drop(columns=no_information_features)

    if len(no_information_features) == 0:
        logger.info('Removed 0 no information features.')
    else:
        logger.info('Removed %d no information features: %s.', len(no_information_features), ', '.join(no_information_features))

    return filtered_df


def remove_redundant_features(df, data_folder, dataset_name, logger, threshold=1.0):
    """
    Removes redundant features with a correlation coefficient of 1.
    """

    # calculate correlation matrix
    corr_matrix = df.corr(numeric_only=True).abs()

    # find features with correlation of at least threshold
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    redundant_features = [col for col in upper.columns if any(upper[col] >= threshold)]

    # remove features with correlation of 1.0
    filtered_df = df.drop(columns=redundant_features)

    if len(redundant_features) == 0:
        logger.info('Removed 0 redundant features.')
    else:
        logger.info('Removed %d redundant features: %s.', len(redundant_features), ', '.join(redundant_features))

    visualize_correlation_matrix(corr_matrix, data_folder, dataset_name, logger)

    return filtered_df


def remove_no_label_rows(df, logger):
    """
    Removes rows without a label.
    """

    # remove rows with missing y
    filtered_df = df.dropna(subset=['y'])

    # count removed rows
    n_total = len(df.index)
    n_removed = n_total - len(filtered_df.index)

    logger.info('Removed %d (%.2f %%) rows without label.', n_removed, n_removed / n_total * 100)

    return filtered_df


def remove_missing_value_rows(df, logger):
    """
    Replace missing values with 'missing' for categoric data and remove rows with numeric missing values.
    """

    # category features with missing values
    missing_value_features = list(df.select_dtypes(include='category').columns[df.select_dtypes(include='category').isna().any()])

    # replace missing values
    replaced_df = df.copy()
    for feature in missing_value_features:
        replaced_df[feature] = replaced_df[feature].cat.add_categories(new_categories='missing')
        replaced_df[feature] = replaced_df[feature].fillna('missing')
    
    if len(missing_value_features) == 0:
        logger.info('Replaced 0 missing values.')
    else:
        logger.info('Replaced missing values in %d categorical features: %s.', len(missing_value_features), ', '.join(missing_value_features))


    missing_value_features_num = list(replaced_df.columns[replaced_df.isna().any()])
    
    filtered_df = replaced_df.dropna(how='any')

    # count removed rows
    n_total = len(df.index)
    n_removed = n_total - len(filtered_df.index)

    logger.info('Removed %d (%.2f %%) rows with missing values in %d features: %s.', n_removed, n_removed / n_total * 100, len(missing_value_features_num), ', '.join(missing_value_features_num))

    return filtered_df


def remove_infinite_value_rows(df, logger):
    """
    Remove rows with infinite values.
    """

    infinite_value_features = list(df.select_dtypes(include='number').columns[df.select_dtypes(include='number').isin([np.inf, -np.inf]).any()])

    filtered_df = df[(~df.select_dtypes(include='number').isin([np.inf, -np.inf])).all(axis='columns')]
    
    # count removed rows
    n_total = len(df.index)
    n_removed = n_total - len(filtered_df.index)

    logger.info('Removed %d (%.2f %%) rows with infinite values in %d features: %s.', n_removed, n_removed / n_total * 100, len(infinite_value_features), ', '.join(infinite_value_features))

    return filtered_df


def remove_negative_value_rows(df, logger):
    """
    Remove rows containing negative values.
    """

    negative_value_features = list(df.select_dtypes(include='number').columns[(df.select_dtypes(include='number') < 0).any()])

    filtered_df = df[(df.select_dtypes(include='number') >= 0).all(axis='columns')]
    
    # count removed rows
    n_total = len(df.index)
    n_removed = n_total - len(filtered_df.index)

    logger.info('Removed %d (%.2f %%) rows with negative values in %d features: %s.', n_removed, n_removed / n_total * 100, len(negative_value_features), ', '.join(negative_value_features))

    return filtered_df


def remove_duplicate_rows(df, data_folder, dataset_name, logger):
    """
    Remove duplicate rows.
    """

    # remove duplicates
    filtered_df = df.drop_duplicates(keep='first')
    filtered_df = filtered_df.drop_duplicates(subset=df.drop(columns='y').columns, keep=False)

    # count removed rows
    n_orig = len(df.index)
    counts_orig = df['y'].value_counts()
    n_filtered = n_orig - len(filtered_df.index)
    counts_filtered = filtered_df['y'].value_counts()

    logger.info('Removed %d (%.2f %%) duplicate rows.', n_filtered, n_filtered / n_orig * 100)
    visualize_duplicates(counts_filtered, counts_orig, data_folder, dataset_name, logger)

    return filtered_df


def remove_too_small_classes(df, logger):
    """
    Removes classes with too few classes. The minimum number of samples is 13, which is calculated by the 60/40 train/test split, 5-fold cross validation and >= 6 samples required for oversampling.
    """

    # min number of rows per class
    min_count = int(np.ceil((1/0.6) * (1/0.8) * 6))

    # count classes
    y_count = df['y'].value_counts()
    too_small_classes = list(y_count[y_count < min_count].index)

    # remove too small classes
    filtered_df = df.copy()[~df['y'].isin(too_small_classes)]
    filtered_df['y'] = filtered_df['y'].cat.remove_categories(removals=too_small_classes)

    if len(too_small_classes) == 0:
        logger.info('Removed 0 classes with too few (< %d) samples.', min_count)
    else:
        logger.info('Removed %d classes with too few (< %d) samples: %s.', len(too_small_classes), min_count, ', '.join(too_small_classes))

    return filtered_df
