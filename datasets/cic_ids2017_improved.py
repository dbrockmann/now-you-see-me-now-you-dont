
import os
import pandas as pd

from utils.data import concat_dataframes


folder_path = 'data-raw/CIC-IDS2017-improved/'
file_names = ['friday.csv', 'monday.csv', 'thursday.csv', 'tuesday.csv', 'wednesday.csv']

names = ['id', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet', 'Total Length of Bwd Packet', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd RST Flags', 'Bwd RST Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'ICMP Code', 'ICMP Type', 'Total TCP Flow Time', 'Label', 'Attempted Category']

remove_names = ['id', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp']
# removing destination port: Establishing the Contaminating Effect of Metadata Feature Inclusion in Machine-Learned Network Intrusion Detection Models
categorical_names = ['Protocol', 'ICMP Code', 'ICMP Type', 'Label']


class CIC_IDS2017_improved:
    """
    Dataset CIC-IDS2017-improved
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
            encoding = 'utf-8'
        ), file_names))

        logger.info('Loaded all %d files.', len(dfs))

        # concat dataframes
        df = concat_dataframes(dfs)

        logger.info('Loaded a total of %d rows.', len(df.index))

        # rename target column
        df = df.rename(columns={'Label': 'y'})

        # rename benign category
        df['y'] = df['y'].cat.rename_categories({'BENIGN': 'Benign'})

        # change attempted attacks to benign, as suggested by the authors
        df.loc[df['Attempted Category'] != -1, 'y'] = 'Benign'
        df = df.drop(columns='Attempted Category')
        df['y'] = df['y'].cat.remove_unused_categories()

        # Note: we keep DoS Hulk, even though it is reported to be wrongly implemented since it still looks anomalous, following the labelling of the improved dataset (ref: https://intrusion-detection.distrinet-research.be/CNS2022/CICIDS2017.html)

        return df
    
dataset = CIC_IDS2017_improved()
