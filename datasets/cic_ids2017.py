
import os
import pandas as pd

from utils.data import concat_dataframes


folder_path = 'data-raw/CIC-IDS2017/'
file_names = ['Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 'Friday-WorkingHours-Morning.pcap_ISCX.csv', 'Monday-WorkingHours.pcap_ISCX.csv', 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv', 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', 'Tuesday-WorkingHours.pcap_ISCX.csv', 'Wednesday-workingHours.pcap_ISCX.csv']

names = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length 2', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']

remove_names = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp', 'Init_Win_bytes_backward', 'Init_Win_bytes_forward'] # Init Win bytes features contain ~50 % negative values, therefore they are removed (Optimizing Cybersecurity Attack Detection in Computer Networks: A Comparative Analysis of Bio-Inspired Optimization Algorithms Using the CSE-CIC-IDS 2018 Dataset)
categorical_names = ['Protocol', 'Label']


class CIC_IDS2017:
    """
    Dataset CIC-IDS2017
    """

    def load(self, logger):
        """"
        Loads the raw dataset and applies first preprocessing steps including type parsing and setting the target column.

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
            encoding = 'utf-8' if not file_name.endswith('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv') else 'latin1' # fix for different encoding of one file
        ), file_names))

        logger.info('Loaded all %d files.', len(dfs))

        # concat dataframes
        df = concat_dataframes(dfs)

        logger.info('Loaded a total of %d rows.', len(df.index))

        # rename target column
        df = df.rename(columns={'Label': 'y'})

        # rename benign category
        df['y'] = df['y'].cat.rename_categories({'BENIGN': 'Benign'})

        return df
    
dataset = CIC_IDS2017()
