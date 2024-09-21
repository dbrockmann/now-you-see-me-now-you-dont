
import os
import pandas as pd

from utils.data import concat_dataframes


folder_path = 'data-raw/uos-dataset/'
file_names = ['uos_dataset_combined.csv']

names = ['uid', 'id.orig_h', 'id.resp_h', 'service', 'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_data_pkts_tot', 'bwd_data_pkts_tot', 'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'flow_pkts_per_sec', 'down_up_ratio', 'fwd_header_size_tot', 'fwd_header_size_min', 'fwd_header_size_max', 'bwd_header_size_tot', 'bwd_header_size_min', 'bwd_header_size_max', 'flow_FIN_flag_count', 'flow_SYN_flag_count', 'flow_RST_flag_count', 'fwd_PSH_flag_count', 'bwd_PSH_flag_count', 'flow_ACK_flag_count', 'fwd_URG_flag_count', 'bwd_URG_flag_count', 'flow_CWR_flag_count', 'flow_ECE_flag_count', 'fwd_pkts_payload.min', 'fwd_pkts_payload.max', 'fwd_pkts_payload.tot', 'fwd_pkts_payload.avg', 'fwd_pkts_payload.std', 'bwd_pkts_payload.min', 'bwd_pkts_payload.max', 'bwd_pkts_payload.tot', 'bwd_pkts_payload.avg', 'bwd_pkts_payload.std', 'flow_pkts_payload.min', 'flow_pkts_payload.max', 'flow_pkts_payload.tot', 'flow_pkts_payload.avg', 'flow_pkts_payload.std', 'fwd_iat.min', 'fwd_iat.max', 'fwd_iat.tot', 'fwd_iat.avg', 'fwd_iat.std', 'bwd_iat.min', 'bwd_iat.max', 'bwd_iat.tot', 'bwd_iat.avg', 'bwd_iat.std', 'flow_iat.min', 'flow_iat.max', 'flow_iat.tot', 'flow_iat.avg', 'flow_iat.std', 'payload_bytes_per_second', 'fwd_subflow_pkts', 'bwd_subflow_pkts', 'fwd_subflow_bytes', 'bwd_subflow_bytes', 'fwd_bulk_bytes', 'bwd_bulk_bytes', 'fwd_bulk_packets', 'bwd_bulk_packets', 'fwd_bulk_rate', 'bwd_bulk_rate', 'active.min', 'active.max', 'active.tot', 'active.avg', 'active.std', 'idle.min', 'idle.max', 'idle.tot', 'idle.avg', 'idle.std', 'fwd_init_window_size', 'bwd_init_window_size', 'fwd_last_window_size', 'bwd_last_window_size', 'attack', 'attack_type', 'ts', 'traffic_direction']

remove_names = ['uid', 'id.orig_h', 'id.resp_h', 'attack', 'ts']
categorical_names = ['attack_type', 'service', 'traffic_direction']


class UOS_IDS23:
    """
    Dataset UOS-IDS23
    """

    def load(self, logger):
        """"
        Loads the raw dataset and applies first preprocessing steps including type parsing.

        Args:
            logger: A logger.

        Returns:
            A dataframe including the entire dataset.
        """

        logger.info('Starting loading of data files.')

        # read csv files
        dfs = list(map(lambda file_name: pd.read_csv(
            os.path.join(folder_path, file_name), 
            header = 0,
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

        # rename target column
        df = df.rename(columns={'attack_type': 'y'})

        # rename benign category
        df['y'] = df['y'].cat.rename_categories({'benign': 'Benign'})

        # merge http/https variants
        def merge_http(label): 
            return label.rsplit('_', 1)[0] if label.endswith('_http') or label.endswith('_https') else label
        df['y'] = df['y'].astype(str)
        df['y'] = df['y'].apply(merge_http)
        df['y'] = pd.Categorical(df['y'])

        return df
    
dataset = UOS_IDS23()
