
import os
import numpy as np
import pandas as pd
import shap

from utils.visualization import visualize_init_success, visualize_class_vulnerability, visualize_mape, visualize_init_success_pert, visualize_shap_values


def evaluate_attacks_init(samples, samples_adv, dataset_names, model_names, max_iters, data_folder, attack_name, logger):
    """
    Evaluate attack initialization using increasing queries
    """

    macro_asr = dict()
    macro_queries = dict()
    for dataset_name in dataset_names:
        macro_asr[dataset_name] = dict()
        macro_queries[dataset_name] = dict()
        for model_name in model_names:
            macro_asr[dataset_name][model_name] = []
            macro_queries[dataset_name][model_name] = []
            for orig, adv in zip(samples[dataset_name][model_name], samples_adv[dataset_name][model_name]):
                asr = get_asr_of_classes(orig, adv)
                queries = get_queries_of_classes(orig, adv)
                macro_asr[dataset_name][model_name].append(np.mean(list(asr.values())))
                macro_queries[dataset_name][model_name].append(np.mean(list(queries.values())))

    visualize_init_success(macro_asr, dataset_names, model_names, max_iters, data_folder, attack_name, logger=logger)

    path = os.path.join(data_folder, 'init_queries_table.txt')
    with open(path, 'w+') as f:
        print(pd.DataFrame(macro_queries).to_string(), file=f, sep='\n')
    logger.info('Saved the initialize attack query table to %s.', path)

def evaluate_attacks_init_pert(samples, samples_adv, dataset_names, model_names, max_pert, data_folder, attack_name, logger):
    """
    Evaluate attack initialization using increasing max perturbation
    """

    macro_asr = dict()
    macro_queries = dict()
    for dataset_name in dataset_names:
        macro_asr[dataset_name] = dict()
        macro_queries[dataset_name] = dict()
        for model_name in model_names:
            macro_asr[dataset_name][model_name] = []
            macro_queries[dataset_name][model_name] = []
            for orig, adv in zip(samples[dataset_name][model_name], samples_adv[dataset_name][model_name]):
                asr = get_asr_of_classes(orig, adv)
                queries = get_queries_of_classes(orig, adv)
                macro_asr[dataset_name][model_name].append(np.mean(list(asr.values())))
                macro_queries[dataset_name][model_name].append(np.mean(list(queries.values())))

    visualize_init_success_pert(macro_asr, dataset_names, model_names, max_pert, data_folder, attack_name, logger=logger)


def get_asr_of_classes(samples, samples_adv):
    """
    Calculate attack success of each class
    """

    classes = np.unique(samples['y'])
    asr = {c: np.mean(samples_adv[samples['y'] == c]['y'] == 'Benign') for c in classes}

    return asr

def get_queries_of_classes(samples, samples_adv):
    """
    Calculate query counts of each class
    """

    classes = np.unique(samples['y'])
    queries = {c: np.mean(samples_adv[samples['y'] == c]['query_count']) for c in classes}

    return queries

def evaluate_attacks_min(samples, samples_adv, features, dataset_names, model_names, attack_names, data_folder, logger):
    """
    Evaluate attack minimizations for comparison
    """

    macro_mape = dict()
    macro_queries = dict()
    for dataset_name in dataset_names:
        macro_mape[dataset_name] = dict()
        macro_queries[dataset_name] = dict()
        for model_name in model_names:
            macro_mape[dataset_name][model_name] = dict()
            macro_queries[dataset_name][model_name] = dict()
            for attack_name in attack_names:
                orig, adv = samples[dataset_name][model_name][attack_name], samples_adv[dataset_name][model_name][attack_name]

                mape = get_mape_of_classes(orig, adv, features[dataset_name])
                queries = get_queries_of_classes(orig, adv)
                macro_mape[dataset_name][model_name][attack_name] = np.mean(list(mape.values()))
                macro_queries[dataset_name][model_name][attack_name] = np.mean(list(queries.values()))

    visualize_mape(macro_mape, macro_queries, dataset_names, model_names, attack_names, data_folder, logger)

    path = os.path.join(data_folder, 'mape_table.txt')
    with open(path, 'w+') as f:
        print(pd.DataFrame(macro_mape).to_string(), file=f, sep='\n')
    logger.info('Saved the MAPE table to %s.', path)

    path = os.path.join(data_folder, 'min_queries_table.txt')
    with open(path, 'w+') as f:
        print(pd.DataFrame(macro_queries).to_string(), file=f, sep='\n')
    logger.info('Saved the minimize attack query table to %s.', path)

def get_mape_of_classes(samples, samples_adv, features):
    """
    Calculate MAPE feature distance between original and adversarial samples
    """

    mape_score = lambda x, x_adv: np.mean(np.abs(np.where(x > 0, (x - x_adv) / (x + 1e-15), 0)), axis=-1)

    classes = np.unique(samples['y'])
    mape = {c: np.mean(mape_score(samples.loc[samples['y'] == c, features].to_numpy(), samples_adv.loc[samples['y'] == c, features].to_numpy())) for c in classes}

    return mape

def evaluate_class_vulnerability(samples_init, samples_init_adv, samples_min, samples_min_adv, model_names, features,  data_folder, dataset_name, attack_name, logger):
    """
    Evaluate the ASR and MAPE of specific classes
    """

    class_asr = None
    class_mape_divider = None
    class_mape = None
    class_mape_divider = None
    for model_name in model_names:
        asr = get_asr_of_classes(samples_init[model_name], samples_init_adv[model_name])
        mape = get_mape_of_classes(samples_min[model_name], samples_min_adv[model_name], features)
        if class_asr is None:
            class_asr = {class_name: 0 for class_name in asr}
            class_asr_divider = class_asr.copy()
            class_mape = class_asr.copy()
            class_mape_divider = class_asr.copy()
        
        for class_name in class_asr:
            if class_name in asr:
                class_asr[class_name] += asr[class_name]
                class_asr_divider[class_name] += 1
            if class_name in mape:
                class_mape[class_name] += mape[class_name]
                class_mape_divider[class_name] += 1

    for class_name in class_asr:
        class_asr[class_name] /= max(class_asr_divider[class_name], 1)
        class_mape[class_name] /= max(class_mape_divider[class_name], 1)

    visualize_class_vulnerability(class_asr, class_mape, data_folder, dataset_name, attack_name, logger)

def evaluate_attack_shap(orig_data, adv_data, model):
    """
    Evaluate perturbations of successful adversarial examples using SHAP
    """

    orig = orig_data.drop(columns='y')
    adv = adv_data.drop(columns=['y', 'query_count'])

    if model['feature_selection'] is not None:
        orig = model['feature_selection'].transform(orig)
        adv = model['feature_selection'].transform(adv)
    orig = model['encoding'].transform(orig)
    orig_columns = orig.columns
    orig = orig.to_numpy()
    adv = model['encoding'].transform(adv).to_numpy()

    expls = []
    for i in range(orig.shape[0]):
        data = np.array([adv[i, :] - orig[i, :]])
        background = np.zeros_like(data)
        explainer = shap.KernelExplainer(model=lambda X: np.array(model['model'].predict_proba(orig[i, :] + X)), data=background)
        expls.append(explainer(data))

    expl = shap.Explanation(
        values=np.concatenate([e.values for e in expls]),
        base_values=expls[0].base_values,
        data=np.concatenate([e.data for e in expls]),
        feature_names=orig_columns
    )

    return expl
