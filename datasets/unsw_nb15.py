
import os
import pandas as pd

from utils.data import concat_dataframes


folder_path = 'data-raw/UNSW-NB15/'
file_names = ['UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 'UNSW-NB15_3.csv', 'UNSW-NB15_4.csv']

names = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label']

remove_names = ['srcip', 'sport', 'dstip', 'dsport', 'Stime', 'Ltime', 'Label', 'ct_ftp_cmd', 'is_ftp_login', 'ct_flw_http_mthd'] # last 3 features mostly NaN and other values than documented (is_ftp_login should be binary)
categorical_names = ['proto', 'state', 'service', 'attack_cat']


class UNSW_NB15:
    """
    Dataset UNSW-NB15
    """

    def load(self, logger):
        """"
        Loads the raw dataset and applies first preprocessing steps including type parsing, assigning the class 'Benign' and merging the classes 'Backdoor' and 'Backdoors'.

        Args:
            logger: A logger.

        Returns:
            A dataframe including the entire dataset.
        """

        logger.info('Starting loading of data files.')

        # read csv files
        dfs = list(map(lambda file_name: pd.read_csv(
            os.path.join(folder_path, file_name), 
            header = None,
            names = names,
            dtype = {**{x: 'category' for x in categorical_names}, **{x: 'float' for x in names if x not in categorical_names + remove_names}},
            usecols = lambda x: x not in remove_names,
            skipinitialspace = True,
            encoding = 'utf-8'
        ), file_names))

        logger.info('Loaded all %d files.', len(dfs))

        # concat dataframes
        df = concat_dataframes(dfs)

        logger.info('Loaded a total of %d rows.', len(df.index))

        # replace NaN values of the feature attack_cat with 'Benign'
        df['attack_cat'] = df['attack_cat'].cat.add_categories('Benign').fillna('Benign')

        # merge attack_cat 'Backdoor' into 'Backdoors'
        df['attack_cat'] = df['attack_cat'].replace(to_replace='Backdoor', value='Backdoors')

        # rename target column
        df = df.rename(columns={'attack_cat': 'y'})

        return df
    
dataset = UNSW_NB15()
