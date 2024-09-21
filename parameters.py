
import numpy as np
from scipy.stats import uniform, loguniform, randint
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from datasets import unsw_nb15, cic_ids2017_improved, cse_cic_ids2018_improved, uos_ids23, BinaryDataset
from models import dnn, cnn, ae, rf, svm, knn
from attacks import RandomInit, BaselineMin, Pointwise, Boundary, HopSkipJump


datasets = {
    'UNSW-NB15': unsw_nb15,
    'CIC-IDS2017-improved': cic_ids2017_improved,
    'CSE-CIC-IDS2018-improved': cse_cic_ids2018_improved,
    'UOS-IDS23': uos_ids23,
    'Bin-UNSW-NB15': BinaryDataset('UNSW-NB15'),
    'Bin-CIC-IDS2017-improved': BinaryDataset('CIC-IDS2017-improved'),
    'Bin-CSE-CIC-IDS2018-improved': BinaryDataset('CSE-CIC-IDS2018-improved'),
    'Bin-UOS-IDS23': BinaryDataset('UOS-IDS23'),
}

models = {
    'DNN': dnn,
    'CNN': cnn,
    'AE': ae,
    'RF': rf,
    'SVM': svm,
    'KNN': knn,
}

model_selection = {
    'n_iter': 20,
    'n_folds': 5,
    'n_jobs': 4,
}

# parameters to fit in randomized search
model_parameters = {
    'DNN': {
        'encoding__normalization': [QuantileTransformer(n_quantiles=1000, random_state=0), MinMaxScaler(clip=True)],
        'model__model__hidden_layer_sizes': [[256], [256, 128], [256, 128, 64]],
        'model__optimizer__learning_rate': loguniform(0.0001, 0.01),
    },
    'CNN': {
        'encoding__normalization': [QuantileTransformer(n_quantiles=1000, random_state=0), MinMaxScaler(clip=True)],
        'model__model__filter_sizes': [[32], [32, 64], [32, 64, 64]],
        'model__model__kernel_size': [3, 5],
        'model__model__hidden_layer_sizes': [[256], [256, 128], [256, 128, 64]],
        'model__model__dropout': uniform(0, 0.5),
        'model__optimizer__learning_rate': loguniform(0.0001, 0.01),
    },
    'AE': {
        'encoding__normalization': [QuantileTransformer(n_quantiles=1000, random_state=0), MinMaxScaler(clip=True)],
        'model__autoencoder__model__hidden_layer_sizes': [[256], [256, 128], [256, 128, 64]],
        'model__autoencoder__model__latent_size': uniform(0.2, 0.6),
        'model__autoencoder__model__l1': loguniform(0.0001, 0.01),
        'model__autoencoder__optimizer__learning_rate': loguniform(0.0001, 0.01),
        'model__classifier__model__hidden_layer_sizes': [[256], [256, 128], [256, 128, 64]],
        'model__classifier__optimizer__learning_rate': loguniform(0.0001, 0.01),
    },
    'RF': {
        'encoding__normalization': [QuantileTransformer(n_quantiles=1000, random_state=0), MinMaxScaler(clip=True)],
        'model__n_estimators': randint(15, 150),
        'model__criterion': ['gini', 'entropy'],
        'model__max_depth': randint(5, 30),
    },
    'SVM': {
        'encoding__normalization': [QuantileTransformer(n_quantiles=1000, random_state=0), MinMaxScaler(clip=True)],
        'model__kernel__gamma': [0.001, 0.1, 1, 10],
        'model__kernel__kernel_params': [{'C': 0.01}, {'C': 0.1}, {'C': 1}, {'C': 10}],
        'model__svm__alpha': loguniform(0.000001, 0.001),
        'model__svm__loss': ['hinge', 'modified_huber'],
    },
    'KNN': {
        'encoding__normalization': [QuantileTransformer(n_quantiles=1000, random_state=0), MinMaxScaler(clip=True)],
        'model__n_neighbors': [5],
    },
}

attacks_init = {
    'RandomInit_1_05': RandomInit(max_iter=1, max_relative_perturbation=0.5),
    'RandomInit_10_05': RandomInit(max_iter=10, max_relative_perturbation=0.5),
    'RandomInit_100_05': RandomInit(max_iter=100, max_relative_perturbation=0.5),
    'RandomInit_1000_05': RandomInit(max_iter=1000, max_relative_perturbation=0.5),

    'RandomInit_10_025': RandomInit(max_iter=10, max_relative_perturbation=0.25),
    'RandomInit_10_075': RandomInit(max_iter=10, max_relative_perturbation=0.75),
    'RandomInit_10_1': RandomInit(max_iter=10, max_relative_perturbation=1),
}

attacks_min = {
    'Baseline': BaselineMin(max_relative_perturbation=0.5),
    'Pointwise': Pointwise(max_relative_perturbation=0.5),
    'Boundary': Boundary(max_relative_perturbation=0.5),
    'HopSkipJump': HopSkipJump(max_relative_perturbation=0.5),
}

features = {
    'UNSW-NB15': {
        'Spkts': {
            'min': 1, # technical min
            'max': 1e+10, # technical max (effective max will be lower due to relative constraint)
            'discrete': True, # discrete/continuous
            'lower': False, # allow lower/max values. default allow
        },
        'sbytes': {
            'min': 0,
            'max': 1e+12,
            'discrete': True,
            'lower': False
        }, 
        'sttl': {
            'min': 10,
            'max': 254,
            'discrete': True
        }, 
    },
    'CIC-IDS2017-improved': {
        'Total Fwd Packet': {
            'min': 1,
            'max': 1e+10,
            'discrete': True,
            'lower': False,
        }, 
        'Total Length of Fwd Packet': {
            'min': 0,
            'max': 1e+12,
            'discrete': True,
            'lower': False,
        }, 
        'Fwd Packet Length Min': {
            'min': 0,
            'max': 1500,
            'discrete': True,
            'higher': False,
        }, 
        'Fwd Packet Length Max': {
            'min': 0,
            'max': 1500,
            'discrete': True,
            'lower': False
        },
    },
    'CSE-CIC-IDS2018-improved': {
        'Total Fwd Packet': {
            'min': 1,
            'max': 1e+10,
            'discrete': True,
            'lower': False,
        }, 
        'Total Length of Fwd Packet': {
            'min': 0,
            'max': 1e+12,
            'discrete': True,
            'lower': False,
        }, 
        'Fwd Packet Length Min': {
            'min': 0,
            'max': 1500,
            'discrete': True,
            'higher': False,
        }, 
        'Fwd Packet Length Max': {
            'min': 0,
            'max': 1500,
            'discrete': True,
            'lower': False
        },
    },
    'UOS-IDS23': {
        'fwd_pkts_tot': {
            'min': 1,
            'max': 1e+10,
            'discrete': True,
            'lower': False,
        },
        'fwd_pkts_payload.tot': {
            'min': 0,
            'max': 1e+12,
            'discrete': True,
            'lower': False,
        }, 
        'fwd_pkts_payload.min': {
            'min': 0,
            'max': 1500,
            'discrete': True,
            'higher': False
        },
        'fwd_pkts_payload.max': {
            'min': 0,
            'max': 1500,
            'discrete': True,
            'lower': False
        },
    }
}

def validate_unsw(X_orig, X):
    """
    Constraints between features for UNSW-NB15
    """

    # constrain total source bytes to max packet length
    X.loc[:, 'sbytes'] = np.clip(X.loc[:, 'sbytes'], a_min=X_orig.loc[:, 'sbytes'], a_max=np.max([X_orig.loc[:, 'sbytes'], X_orig.loc[:, 'Spkts'] * 1500], axis=0) + (X.loc[:, 'Spkts'] - X_orig.loc[:, 'Spkts']) * 1500)

    # update load and packages per second as well as mean size of packets based on sbytes
    X.loc[:, 'Sload'] = np.where(X.loc[:, 'dur'] == 0, 0, X.loc[:, 'sbytes'] * 8 / X.loc[:, 'dur'])
    X.loc[:, 'smeansz'] = X.loc[:, 'sbytes'] / X.loc[:, 'Spkts']
    X.loc[:, 'Sintpkt'] = np.where(X.loc[:, 'Spkts'] <= 1, 0, X.loc[:, 'dur'] / (X.loc[:, 'Spkts'] - 1) * 1e+03)

    return X

def validate_ids_improved(X_orig, X):
    """
    Constraints between features for improved versions of CIC-IDS2017 and CSE-CIC-IDS2018
    """

    # enforce constraints between min and max (should not be needed, as min/max can only be lower/higher)
    X.loc[:, 'Fwd Packet Length Min'] = np.clip(X.loc[:, 'Fwd Packet Length Min'], a_min=None, a_max=X.loc[:, 'Fwd Packet Length Max'])
    X.loc[:, 'Fwd Packet Length Max'] = np.max([X.loc[:, 'Fwd Packet Length Max'], X_orig.loc[:, 'Fwd Packet Length Max']], axis=0)

    # minimum number of packets based on min/max changes: for a lower min, an additional packet needs to be added; for different min/max, at least 2 packets need to be present
    X.loc[:, 'Total Fwd Packet'] = np.clip(X.loc[:, 'Total Fwd Packet'], a_min=np.where(X.loc[:, 'Fwd Packet Length Min'] == X.loc[:, 'Fwd Packet Length Max'], 1, 2), a_max=None)
    X.loc[:, 'Total Fwd Packet'] = np.clip(X.loc[:, 'Total Fwd Packet'], a_min=np.where(X.loc[:, 'Fwd Packet Length Min'] < X_orig.loc[:, 'Fwd Packet Length Min'], X_orig.loc[:, 'Total Fwd Packet'] + 1, X_orig.loc[:, 'Total Fwd Packet']), a_max=None)
    X.loc[:, 'Fwd Header Length'] = np.clip(X.loc[:, 'Fwd Header Length'], X.loc[:, 'Total Fwd Packet'] * 20, X.loc[:, 'Total Fwd Packet'] * 60)

    # update total length based on added packets and update mean length based on that
    added_pkts = X.loc[:, 'Total Fwd Packet'] - X_orig.loc[:, 'Total Fwd Packet']
    X.loc[:, 'Total Length of Fwd Packet'] = np.clip(
        X.loc[:, 'Total Length of Fwd Packet'], 
        a_min=X_orig.loc[:, 'Total Length of Fwd Packet'] + added_pkts * X.loc[:, 'Fwd Packet Length Min'] + (X.loc[:, 'Fwd Packet Length Max'] - X_orig.loc[:, 'Fwd Packet Length Max']), 
        a_max=np.max([X_orig.loc[:, 'Total Length of Fwd Packet'], X_orig.loc[:, 'Total Fwd Packet'] * 1500], axis=0) + (added_pkts - 1) * X.loc[:, 'Fwd Packet Length Max'] + np.where(X.loc[:, 'Fwd Packet Length Min'] < X_orig.loc[:, 'Fwd Packet Length Min'], X.loc[:, 'Fwd Packet Length Min'], X.loc[:, 'Fwd Packet Length Max']))
    X.loc[:, 'Fwd Packet Length Mean'] = X.loc[:, 'Total Length of Fwd Packet'] / X.loc[:, 'Total Fwd Packet']

    # recalculate per second features
    X.loc[:, 'Flow Bytes/s'] = (X.loc[:, 'Total Length of Fwd Packet'] + X.loc[:, 'Total Length of Bwd Packet']) / (X.loc[:, 'Flow Duration'] * 1e-06)
    X.loc[:, 'Flow Packets/s'] = (X.loc[:, 'Total Fwd Packet'] + X.loc[:, 'Total Bwd packets']) / (X.loc[:, 'Flow Duration'] * 1e-06)

    # update IAT features
    X.loc[:, 'Fwd IAT Mean'] = np.where(X.loc[:, 'Total Fwd Packet'] <= 1, 0, X.loc[:, 'Fwd IAT Total'] / (X.loc[:, 'Total Fwd Packet'] - 1))
    X.loc[:, 'Fwd IAT Min'] = np.clip(X.loc[:, 'Fwd IAT Min'], a_min=None, a_max=X.loc[:, 'Fwd IAT Mean'])
    X.loc[:, 'Fwd IAT Max'] = np.clip(X.loc[:, 'Fwd IAT Max'], a_min=X.loc[:, 'Fwd IAT Mean'], a_max=None)
    min_max_dist = np.square(X.loc[:, 'Fwd IAT Min'] - X.loc[:, 'Fwd IAT Mean']) + np.square(X.loc[:, 'Fwd IAT Max'] - X.loc[:, 'Fwd IAT Mean'])
    X.loc[:, 'Fwd IAT Std'] = np.clip(X.loc[:, 'Fwd IAT Std'], np.where(X.loc[:, 'Total Fwd Packet'] == 1, 0, np.sqrt(min_max_dist / X.loc[:, 'Total Fwd Packet'])), np.sqrt(min_max_dist / 2))

    X.loc[:, 'Flow IAT Min'] = np.min([X.loc[:, 'Flow IAT Min'], X.loc[:, 'Fwd IAT Min']], axis=0)
    X.loc[:, 'Flow IAT Mean'] = np.where(X.loc[:, 'Total Fwd Packet'] + X.loc[:, 'Total Bwd packets'] <= 1, 0, X.loc[:, 'Flow Duration'] / (X.loc[:, 'Total Fwd Packet'] + X.loc[:, 'Total Bwd packets'] - 1))

    # update global metrics and down/up ratio; note, down/up ratio was rounded in the original IDS17/18 dataset but are corrected for the improved versions
    X.loc[:, 'Packet Length Min'] = np.min([X.loc[:, 'Packet Length Min'], X.loc[:, 'Fwd Packet Length Min']], axis=0)
    X.loc[:, 'Packet Length Mean'] = np.where(X.loc[:, 'Total Fwd Packet'] + X.loc[:, 'Total Bwd packets'] == 0, 0, (X.loc[:, 'Total Length of Fwd Packet'] + X.loc[:, 'Total Length of Bwd Packet']) / (X.loc[:, 'Total Fwd Packet'] + X.loc[:, 'Total Bwd packets']))
    X.loc[:, 'Down/Up Ratio'] = np.where(X.loc[:, 'Total Fwd Packet'] == 0, 0, X.loc[:, 'Total Bwd packets'] / X.loc[:, 'Total Fwd Packet'])

    return X

def validate_uos(X_orig, X):
    """
    Constraints between features for UOS-IDS23
    """

    # generally equal to ids-improved but with different feature names

    # enforce constraints between min and max (should not be needed, as min/max can only be lower/higher)
    X.loc[:, 'fwd_pkts_payload.min'] = np.clip(X.loc[:, 'fwd_pkts_payload.min'], a_min=None, a_max=X.loc[:, 'fwd_pkts_payload.max'])
    X.loc[:, 'fwd_pkts_payload.max'] = np.max([X.loc[:, 'fwd_pkts_payload.max'], X_orig.loc[:, 'fwd_pkts_payload.max']], axis=0)

    # minimum number of packets based on min/max changes: for a lower min, an additional packet needs to be added; for different min/max, at least 2 packets need to be present
    X.loc[:, 'fwd_pkts_tot'] = np.clip(X.loc[:, 'fwd_pkts_tot'], a_min=np.where(X.loc[:, 'fwd_pkts_payload.min'] == X.loc[:, 'fwd_pkts_payload.max'], 1, 2), a_max=None)
    X.loc[:, 'fwd_pkts_tot'] = np.clip(X.loc[:, 'fwd_pkts_tot'], a_min=np.where(X.loc[:, 'fwd_pkts_payload.min'] < X_orig.loc[:, 'fwd_pkts_payload.min'], X_orig.loc[:, 'fwd_pkts_tot'] + 1, X_orig.loc[:, 'fwd_pkts_tot']), a_max=None)
    X.loc[:, 'fwd_header_size_tot'] = np.clip(X.loc[:, 'fwd_header_size_tot'], a_min=(X.loc[:, 'fwd_pkts_tot'] - 1) * X.loc[:, 'fwd_header_size_min'] + X.loc[:, 'fwd_header_size_max'], a_max=(X.loc[:, 'fwd_pkts_tot'] - 1) * X.loc[:, 'fwd_header_size_max'] + X.loc[:, 'fwd_header_size_min'])

    # update total length based on added packets and update mean length based on that
    added_pkts = X.loc[:, 'fwd_pkts_tot'] - X_orig.loc[:, 'fwd_pkts_tot']
    X.loc[:, 'fwd_pkts_payload.tot'] = np.clip(
        X.loc[:, 'fwd_pkts_payload.tot'], 
        a_min=X_orig.loc[:, 'fwd_pkts_payload.tot'] + added_pkts * X.loc[:, 'fwd_pkts_payload.min'] + (X.loc[:, 'fwd_pkts_payload.max'] - X_orig.loc[:, 'fwd_pkts_payload.max']), 
        a_max=np.max([X_orig.loc[:, 'fwd_pkts_payload.tot'], X_orig.loc[:, 'fwd_pkts_tot'] * 1500], axis=0) + (added_pkts - 1) * X.loc[:, 'fwd_pkts_payload.max'] + np.where(X.loc[:, 'fwd_pkts_payload.min'] < X_orig.loc[:, 'fwd_pkts_payload.min'], X.loc[:, 'fwd_pkts_payload.min'], X.loc[:, 'fwd_pkts_payload.max']))
    X.loc[:, 'fwd_pkts_payload.avg'] = X.loc[:, 'fwd_pkts_payload.tot'] / X.loc[:, 'fwd_pkts_tot']

    # recalculate per second features
    X.loc[:, 'fwd_pkts_per_sec'] = np.where(X.loc[:, 'flow_duration'] == 0.0, 0.0, X.loc[:, 'fwd_pkts_tot'] / X.loc[:, 'flow_duration'])
    X.loc[:, 'payload_bytes_per_second'] = np.where(X.loc[:, 'flow_duration'] == 0.0, 0.0, (X.loc[:, 'fwd_pkts_payload.tot'] + X.loc[:, 'bwd_pkts_payload.tot']) / X.loc[:, 'flow_duration'])

    # update IAT features
    X.loc[:, 'fwd_iat.avg'] = np.where(X.loc[:, 'fwd_pkts_tot'] <= 1, 0, X.loc[:, 'fwd_iat.tot'] / (X.loc[:, 'fwd_pkts_tot'] - 1))
    X.loc[:, 'fwd_iat.min'] = np.clip(X.loc[:, 'fwd_iat.min'], a_min=None, a_max=X.loc[:, 'fwd_iat.avg'])
    X.loc[:, 'fwd_iat.max'] = np.clip(X.loc[:, 'fwd_iat.max'], a_min=X.loc[:, 'fwd_iat.avg'], a_max=None)
    min_max_dist = np.square(X.loc[:, 'fwd_iat.min'] - X.loc[:, 'fwd_iat.avg']) + np.square(X.loc[:, 'fwd_iat.max'] - X.loc[:, 'fwd_iat.avg'])
    X.loc[:, 'fwd_iat.std'] = np.clip(X.loc[:, 'fwd_iat.std'], np.where(X.loc[:, 'fwd_pkts_tot'] == 1, 0, np.sqrt(min_max_dist / X.loc[:, 'fwd_pkts_tot'])), np.sqrt(min_max_dist / 2))

    X.loc[:, 'flow_iat.min'] = np.min([X.loc[:, 'flow_iat.min'], X.loc[:, 'fwd_iat.min']], axis=0)
    X.loc[:, 'flow_iat.avg'] = np.where(X.loc[:, 'fwd_pkts_tot'] + X.loc[:, 'bwd_pkts_tot'] <= 1, 0, X.loc[:, 'flow_duration'] / (X.loc[:, 'fwd_pkts_tot'] + X.loc[:, 'bwd_pkts_tot'] - 1))

    # update global metrics and down/up ratio
    X.loc[:, 'flow_pkts_payload.min'] = np.min([X.loc[:, 'flow_pkts_payload.min'], X.loc[:, 'fwd_pkts_payload.min']], axis=0)
    X.loc[:, 'flow_pkts_payload.avg'] = np.where(X.loc[:, 'fwd_pkts_tot'] + X.loc[:, 'bwd_pkts_tot'] == 0, 0, (X.loc[:, 'fwd_pkts_payload.tot'] + X.loc[:, 'bwd_pkts_payload.tot']) / (X.loc[:, 'fwd_pkts_tot'] + X.loc[:, 'bwd_pkts_tot']))
    X.loc[:, 'down_up_ratio'] = np.where(X.loc[:, 'fwd_pkts_tot'] == 0, 0, X.loc[:, 'bwd_pkts_tot'] / X.loc[:, 'fwd_pkts_tot'])

    # other impacted features are removed in the feature selection; for simplicity and since they are not propagated to the classifier anyway, we do not include all of those here

    return X

feature_validation = {
    'UNSW-NB15': validate_unsw,
    'CIC-IDS2017-improved': validate_ids_improved,
    'CSE-CIC-IDS2018-improved': validate_ids_improved,
    'UOS-IDS23': validate_uos
}

for dataset_name in list(features):
    features[f'Bin-{dataset_name}'] = features[dataset_name]
for dataset_name in list(feature_validation):
    feature_validation[f'Bin-{dataset_name}'] = feature_validation[dataset_name]
