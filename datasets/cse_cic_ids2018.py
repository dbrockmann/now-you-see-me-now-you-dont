
import os
import pandas as pd

from utils.data import concat_dataframes


folder_path = 'data-raw/CSE-CIC-IDS2018/'
file_names = ['Friday-02-03-2018_TrafficForML_CICFlowMeter.csv', 'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv', 'Friday-23-02-2018_TrafficForML_CICFlowMeter.csv', 'Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv', 'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv', 'Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv', 'Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv', 'Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv', 'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv', 'Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv']

names = ['Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']

remove_names = ['Dst Port', 'Timestamp', 'Init Bwd Win Byts', 'Init Fwd Win Byts'] # Init Win bytes features contain ~50 % negative values, therefore they are removed (Optimizing Cybersecurity Attack Detection in Computer Networks: A Comparative Analysis of Bio-Inspired Optimization Algorithms Using the CSE-CIC-IDS 2018 Dataset)
categorical_names = ['Protocol', 'Label']

# fix to ignore additional header rows in the data
additional_header_rows = {
    'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv': [1000000],
    'Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv': [414, 19762, 19907, 39020, 60810, 76529, 81060, 85449, 89954, 91405, 92658, 95061, 331113, 331114, 331115, 331116, 331117, 331118, 331119, 331120, 331121, 331122, 331123, 331124, 331125],
    'Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv': [21839, 43118, 63292, 84014, 107720, 132410, 154206, 160207, 202681, 228584, 247718, 271677, 296995, 322939, 344163, 349510, 355080, 360661, 366040, 367414, 368614, 371160, 377705, 399544, 420823, 440997, 461719, 485425, 510115, 534074, 559392, 585336, 606560]
}

class CSE_CIC_IDS2018:
    """
    Dataset CSE-CIC-IDS2018
    """

    def load(self, logger):
        """"
        Loads the raw dataset and applies first preprocessing steps including type parsing and the removal of additional header rows in the CSV files.

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
            encoding = 'utf-8',
            skiprows = [] if file_name not in additional_header_rows.keys() else additional_header_rows[file_name]
        ), file_names))

        logger.info('Loaded all %d files.', len(dfs))

        # concat dataframes
        df = concat_dataframes(dfs)

        logger.info('Loaded a total of %d rows.', len(df.index))

        # rename target column
        df = df.rename(columns={'Label': 'y'})

        return df
    
dataset = CSE_CIC_IDS2018()
