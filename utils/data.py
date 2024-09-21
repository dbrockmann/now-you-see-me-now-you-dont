
import re
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_categories(df):
    """
    Cleans category names by removing doubled, leading and trailing spaces and special characters.

    Args:
        df: A dataframe.

    Returns:
        The dataframe with cleaned categories. 
    """

    # regular expression to remove additional spaces and special characters
    clean_str = lambda x: re.sub(r' {2,}', ' ', re.sub(r'[^\w\s]', '', x.strip())) if isinstance(x, str) else x

    # apply regex to all category columns
    for name in df.select_dtypes(include='category').columns:
        cleaned_data = df[name].map(clean_str)
        cleaned_categories = set(df[name].cat.categories.map(clean_str))
        df[name] = pd.Categorical(cleaned_data, categories=cleaned_categories)

    return df

def concat_dataframes(dfs):
    """
    Concat dataframes by first cleaning and merging the categories.

    Args:
        dfs: A list of dataframes.

    Returns:
        A dataframe combining all dataframes.
    """

    # clean categories for each dataframe
    for i in range(len(dfs)):
        dfs[i] = clean_categories(dfs[i])

    # create new categoricals to include the categories of all dataframes
    for name in dfs[0].select_dtypes(include='category').columns:
        union_categories = set().union(*[df[name].cat.categories for df in dfs])
        for df in dfs:
            df[name] = pd.Categorical(df[name], categories=union_categories)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.reset_index(drop=True, inplace=True)

    return combined_df

def split_data(df, split=0.5, logger=None):
    """
    Splits a dataframe into two partitions with the specified split. Additionally, the dataframe is shuffled prior to the split and the split is stratified along the class column to keep the class ratios.

    Args:
        split: The split ratio given by the size of the second set.
        logger: A logger.

    Returns:
        Two dataframes.
    """

    

    df_1, df_2 = train_test_split(df, test_size=split, shuffle=True, stratify=df['y'], random_state=0)
    df_1 = df_1.reset_index(drop=True)
    df_2 = df_2.reset_index(drop=True)

    if logger is not None:
        logger.info('Splitted the dataset into two partitions with a split of %.2f, resulting in %d and %d rows.', split, len(df_1.index), len(df_2.index))

    return df_1, df_2

def take_n_per_class(df, n=1000):
    """
    Sample up to n samples per class
    """

    sampled_df = df.groupby('y').apply(lambda x: x.sample(n=min(n, len(x)), replace=False, random_state=0)).reset_index(drop=True)

    return sampled_df
