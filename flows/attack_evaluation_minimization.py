
from parameters import datasets, models, attacks_init, attacks_min, features
from attacking.attack_evaluation import evaluate_attack_shap, evaluate_attacks_min, evaluate_class_vulnerability
from utils.visualization import visualize_attack_features, visualize_shap_values
from utils.io import get_logger, load_model, load_samples


def attack_evaluation_minimization(data_folder):
    """
    Evaluate adversarial examples with minimization
    """

    flow_logger = get_logger('attack-evaluation-minimization')
    flow_logger.info('Starting minimization attack evaluation for the following datasets: %s; the following models: %s; and the following minimization attacks: %s.', ', '.join(datasets.keys()), ', '.join(models.keys()), ', '.join(attacks_min.keys()))

    samples, samples_adv = dict(), dict()

    explanations = dict()
    labels = dict()

    for dataset_name in datasets:
        samples[dataset_name], samples_adv[dataset_name] = dict(), dict()
        samples_init, samples_adv_init = dict(), dict()

        for model_name in models:
            samples[dataset_name][model_name], samples_adv[dataset_name][model_name] = dict(), dict()
            samples_init[model_name], samples_adv_init[model_name] = dict(), dict()

            model = load_model(data_folder, dataset_name, model_name, logger=flow_logger)

            for attack_name_init in attacks_init:
                samples_init[model_name][attack_name_init], samples_adv_init[model_name][attack_name_init] = load_samples(data_folder, dataset_name, model_name, attack_name_init, logger=flow_logger)

                for attack_name in attacks_min:

                    # load samples
                    orig, adv = load_samples(data_folder, dataset_name, model_name, f'{attack_name_init}_{attack_name}', logger=flow_logger)

                    init_samples, init_samples_adv = load_samples(data_folder, dataset_name, model_name, attack_name_init, logger=flow_logger)
                    unsuc_samples = init_samples.loc[init_samples_adv['y'] != 'Benign', :].reset_index(drop=True)

                    visualize_attack_features(orig, adv, unsuc_samples, list(features[dataset_name]), data_folder, dataset_name, model_name, f'{attack_name_init}_{attack_name}', flow_logger)

                    samples[dataset_name][model_name][f'{attack_name_init}_{attack_name}'] = orig
                    samples_adv[dataset_name][model_name][f'{attack_name_init}_{attack_name}'] = adv

                    explanations[dataset_name] = evaluate_attack_shap(orig, adv, model)
                    labels[dataset_name] = model['model'].classes_

        for attack_name_init in attacks_init:
            for attack_name in attacks_min:
                evaluate_class_vulnerability({model_name: samples_init[model_name][attack_name_init] for model_name in models}, {model_name: samples_adv_init[model_name][attack_name_init] for model_name in models}, {model_name: samples[dataset_name][model_name][f'{attack_name_init}_{attack_name}'] for model_name in models}, {model_name: samples_adv[dataset_name][model_name][f'{attack_name_init}_{attack_name}'] for model_name in models}, list(models), list(features[dataset_name]), data_folder, dataset_name, f'{attack_name_init}_{attack_name}', flow_logger)

    attack_names = [f'{attack_name_init}_{attack_name}' for attack_name_init in attacks_init for attack_name in attacks_min]
    evaluate_attacks_min(samples, samples_adv, {d: list(features[d]) for d in datasets}, list(datasets), list(models), attack_names, data_folder, flow_logger)

    visualize_shap_values(explanations=explanations, labels=labels, data_folder=data_folder, dataset_names=list(datasets), logger=flow_logger)
