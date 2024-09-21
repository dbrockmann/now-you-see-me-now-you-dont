
import os
import textwrap
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import MinMaxScaler
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import classification_report, confusion_matrix
matplotlib.use('Agg')

import scienceplots
plt.style.use(['science', 'ieee', 'high-contrast'])
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['palatino'],
})


def visualize_correlation_matrix(correlation_matrix, data_folder, dataset_name, logger):
    """
    Visualizes a correlation matrix.
    """

    path = os.path.join(data_folder, dataset_name, f'{dataset_name}_correlation-matrix_table.txt')
    with open(path, 'w+') as f:
        print(correlation_matrix.to_string(), file=f)
    logger.info('Saved the correlation matrix table to %s.', path)

    fig = plt.figure(figsize=(24, 20), dpi=100)
    ax = fig.gca()
    mat_ax = ax.matshow(correlation_matrix, cmap=sns.color_palette('rocket', as_cmap=True), interpolation='nearest')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.5)
    cb = fig.colorbar(mat_ax, cax=cax)

    labels = correlation_matrix.columns
    xaxis = np.arange(len(labels))
    ax.xaxis.tick_bottom()
    ax.set_xticks(xaxis)
    ax.set_yticks(xaxis)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.tick_params('x', top=None, bottom=None, which='both', labelsize=18)
    ax.tick_params('y', left=None, right=None, which='both', labelsize=18)

    ax.set_xlabel('Feature', fontsize=26)
    ax.set_ylabel('Feature', fontsize=26)

    cb.ax.tick_params('y', length=6, width=2, which='major', labelsize=26)
    cb.ax.tick_params('y', length=4, width=1, which='minor')
    cb.ax.set_ylabel('Absolute Correlation Coefficient', fontsize=26)

    path = os.path.join(data_folder, dataset_name, f'{dataset_name}_correlation-matrix_plot.png')
    plt.savefig(path)
    logger.info('Saved the correlation matrix plot to %s.', path)

def visualize_duplicates(counts_filtered, counts_orig, data_folder, dataset_name, logger):
    """
    Visualizes the number of duplicates in each class.
    """

    counts_duplicates = pd.DataFrame({'duplicates': counts_orig - counts_filtered})
    counts_duplicates['duplicates_share'] = counts_duplicates['duplicates'] / counts_orig
    counts_duplicates = counts_duplicates.sort_values(by='duplicates_share', ascending=False)

    path = os.path.join(data_folder, dataset_name, f'{dataset_name}_duplicates_table.txt')
    with open(path, 'w+') as f:
        print(counts_duplicates.to_string(), file=f)
    logger.info('Saved the duplicates table to %s.', path)

    ax = counts_duplicates.plot.bar(y=['duplicates_share'], legend=False, figsize=(8, 6), fontsize=20, width=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    plt.xlabel('Class', fontsize=20)
    plt.ylabel('Percentage of Duplicates', fontsize=20)
    ax.tick_params('x', top=None, bottom=None, which='both', labelsize=20)
    ax.tick_params('y', length=6, width=2, which='major', labelsize=20)
    ax.tick_params('y', length=4, width=1, which='minor')

    path = os.path.join(data_folder, dataset_name, f'{dataset_name}_duplicates_plot.png')
    plt.savefig(path)
    logger.info('Saved the duplicates plot to %s.', path)

def visualize_undersampling_test(scores, n_samples, data_folder, dataset_name, logger):
    """
    Visualize the undersampling optimization
    """

    path = os.path.join(data_folder, dataset_name, f'{dataset_name}_undersampling_test_table.txt')
    with open(path, 'w+') as f:
        print(pd.DataFrame(scores).to_string(), n_samples, file=f, sep='\n')
    logger.info('Saved the undersampling test table to %s.', path)

    plt.figure(figsize=(10, 6), dpi=100)

    for model_name in scores:
        plt.plot(n_samples, scores[model_name]['test_mean'], label=f'{model_name} Test', linewidth=3.0)
        plt.fill_between(n_samples, np.array(scores[model_name]['test_mean']) + scores[model_name]['test_std'], np.array(scores[model_name]['test_mean']) - scores[model_name]['test_std'], alpha=0.1)

    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.tick_params('y', length=6, width=2, which='major', labelsize=20)
    ax.tick_params('y', length=4, width=1, which='minor')
    ax.tick_params('x', top=None, length=6, width=2, which='major', labelsize=20)
    ax.tick_params('x', top=None, length=6, width=2, which='minor', labelsize=20)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xscale('log', base=10)
    ax.xaxis.set_major_formatter(lambda x, pos: f'{x}')
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(n_samples[::2]))
    ax.xaxis.set_minor_formatter(lambda x, pos: f'\n{x}')
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(n_samples[1::2]))
    ax.invert_xaxis()
    plt.xlabel('Number of Training Samples', fontsize=20)
    plt.ylabel('Macro F1', fontsize=20)
    
    path = os.path.join(data_folder, dataset_name, f'{dataset_name}_undersampling_test_plot.png')
    plt.savefig(path)
    logger.info('Saved the undersampling test plot to %s.', path)

def visualize_oversampling_test(scores, n_samples, data_folder, dataset_name, class_name, logger):
    """
    Visualize the oversampling optimization
    """

    path = os.path.join(data_folder, dataset_name, f'{dataset_name}_{class_name}_oversampling_test_table.txt')
    with open(path, 'w+') as f:
        print(pd.DataFrame(scores).to_string(), n_samples, file=f, sep='\n')
    logger.info('Saved the oversampling test table to %s.', path)

    plt.figure(figsize=(10, 6), dpi=100)

    for model_name in scores:
        plt.plot(n_samples, scores[model_name]['test_mean'], label=f'{model_name} Test', linewidth=3.0)
        plt.fill_between(n_samples, np.array(scores[model_name]['test_mean']) + scores[model_name]['test_std'], np.array(scores[model_name]['test_mean']) - scores[model_name]['test_std'], alpha=0.1)
        plt.plot(n_samples, scores[model_name]['test_mean_class'], label=f'{model_name} {class_name} Test', linewidth=3.0, linestyle=':')
        plt.fill_between(n_samples, np.array(scores[model_name]['test_mean_class']) + scores[model_name]['test_std_class'], np.array(scores[model_name]['test_mean_class']) - scores[model_name]['test_std_class'], alpha=0.1)

    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.tick_params('y', length=6, width=2, which='major', labelsize=20)
    ax.tick_params('y', length=4, width=1, which='minor')
    ax.tick_params('x', top=None, length=6, width=2, which='major', labelsize=20)
    ax.tick_params('x', top=None, bottom=None, which='minor')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xscale('log', base=10)
    ax.xaxis.set_major_formatter(lambda x, pos: f'{x}')
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(n_samples))
    plt.xlabel(f'Number of {class_name} Training Samples', fontsize=20)
    plt.ylabel('Macro F1', fontsize=20)
    
    path = os.path.join(data_folder, dataset_name, f'{dataset_name}_{class_name}_oversampling_test_plot.png')
    plt.savefig(path)
    logger.info('Saved the oversampling test plot to %s.', path)

def visualize_feature_selection_test(scores, thresholds, n_features, data_folder, dataset_name, logger):
    """
    Visualize feature selection optimization
    """

    path = os.path.join(data_folder, dataset_name, f'{dataset_name}_feature_selection_test_table.txt')
    with open(path, 'w+') as f:
        print(pd.DataFrame(scores).to_string(), thresholds, n_features, file=f, sep='\n')
    logger.info('Saved the feature selection test table to %s.', path)

    plt.figure(figsize=(9, 8), dpi=100)
    ax = plt.gca()

    markers = ['x', 's', '*']
    for model_name, marker in zip(scores, markers):
        sc = ax.scatter(n_features, scores[model_name]['test_mean'], label=f'{model_name} Test', marker=marker, s=100)
        for x, y, t in zip(n_features, scores[model_name]['test_mean'], thresholds):
            ax.annotate(f'{t:.1f}', xy=(x, y), textcoords='offset pixels', xytext=(0, 10), ha='center', fontsize=16, color=sc.get_facecolor())

    ax.set_axisbelow(True)
    ax.tick_params('y', length=6, width=2, which='major', labelsize=20)
    ax.tick_params('y', length=4, width=1, which='minor')
    ax.tick_params('x', top=None, length=6, width=2, which='major', labelsize=20)
    ax.tick_params('x', bottom=None, top=None, which='minor')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))

    ax.xaxis.set_major_formatter(lambda x, pos: f'{x}')
    ax.set_xlim(n_features[0]+(n_features[0]-n_features[-1])*0.075, n_features[-1]-(n_features[0]-n_features[-1])*0.075)
    plt.xticks(np.linspace(n_features[0], n_features[-1], num=6, dtype=int))
    plt.xlabel('Number of Features', fontsize=20)
    plt.ylabel('Macro F1', fontsize=20)
    
    path = os.path.join(data_folder, dataset_name, f'{dataset_name}_feature_selection_test_plot.png')
    plt.savefig(path)
    logger.info('Saved the feature selection test plot to %s.', path)

def visualize_retraining(history, data_folder, dataset_name, model_name, logger):
    """
    Visualize training curves
    """

    path = os.path.join(data_folder, dataset_name, model_name, f'{dataset_name}_{model_name}_retraining_table.txt')
    with open(path, 'w+') as f:
        print(pd.DataFrame(history).to_string(), file=f)
    logger.info('Saved the retraining table to %s.', path)

    plt.figure(figsize=(10, 6), dpi=100)

    epochs = np.arange(len(history['loss']))
    loss_line, = plt.plot(epochs, history['loss'], label='Train Loss', linewidth=3.0)
    plt.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=3.0, color=loss_line.get_color(), linestyle='--')
    acc_line, = plt.plot(epochs, history['sparse_categorical_accuracy'], label='Train Accuracy', linewidth=3.0)
    plt.plot(epochs, history['val_sparse_categorical_accuracy'], label='Validation Accuracy', linewidth=3.0, color=acc_line.get_color(), linestyle='--')

    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.tick_params('y', length=6, width=2, which='major', labelsize=20)
    ax.tick_params('y', length=4, width=1, which='minor')
    ax.tick_params('x', top=None, length=6, width=2, which='major', labelsize=20)
    ax.tick_params('x', top=None, bottom=None, which='minor')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel('Training Epoch', fontsize=20)
    plt.ylabel('Metric', fontsize=20)
    
    path = os.path.join(data_folder, dataset_name, model_name, f'{dataset_name}_{model_name}_retraining_plot.png')
    plt.savefig(path)
    logger.info('Saved the retraining plot to %s.', path)

def visualize_class_shares(class_counts, data_folder, dataset_name, model_name, logger):
    """
    Visualize class shares for undersampling and oversampling
    """

    class_shares = class_counts.sort_values(by=class_counts.columns[0], ascending=False).astype('int')

    for i, column in enumerate(class_shares.columns):
        class_shares.insert(i * 2 + 1, f'{column}_share', class_shares[column] / class_shares[column].sum())

    path = os.path.join(data_folder, dataset_name, model_name, f'{dataset_name}_{model_name}_class_shares.txt')
    with open(path, 'w+') as f:
        print(class_shares, file=f)
    logger.info('Saved the class shares table to %s.', path)

    ax = class_shares.plot.bar(y=[column for column in class_shares.columns if not column.endswith('_share')], logy=True, figsize=(8, 6), fontsize=20, width=0.8)
    ax.set_axisbelow(True)
    plt.xlabel('Class', fontsize=20)
    plt.ylabel('Number of Samples', fontsize=20)
    plt.legend(fontsize=20)
    ax.tick_params('x', top=None, bottom=None, which='both', labelsize=20)
    ax.tick_params('y', length=8, width=2, which='major', labelsize=20)
    ax.tick_params('y', length=4, width=1, which='minor')

    path = os.path.join(data_folder, dataset_name, model_name, f'{dataset_name}_{model_name}_class_shares.png')
    plt.savefig(path)
    logger.info('Saved the class shares plot to %s.', path)

def visualize_classification_report(y_true, y_pred, labels, data_folder, dataset_name, model_name, logger):
    """
    Save the classification report
    """

    report = classification_report(y_true=y_true, y_pred=y_pred, labels=labels, digits=4)

    path = os.path.join(data_folder, dataset_name, model_name, f'{dataset_name}_{model_name}_classification_report.txt')
    with open(path, 'w+') as f:
        print(report, file=f)
    logger.info('Saved the classification report to %s.', path)

def visualize_confusion_matrix(y_true, y_pred, labels, data_folder, dataset_name, model_name, logger):
    """
    Visualize a confusion matrix
    """

    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    conf_matrix_norm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels, normalize='true')
    conf_annot = np.reshape([f'{share:.2f}' for total, share in zip(conf_matrix.flatten(), conf_matrix_norm.flatten())], conf_matrix.shape)
    conf_annot = np.vectorize(lambda x: '1' if x == '1.00' else '' if x == '0.00' else f'.{x.split(".")[-1]}')(conf_annot)

    conf_annot_print = np.reshape([f'{total} ({share:.4f})' for total, share in zip(conf_matrix.flatten(), conf_matrix_norm.flatten())], conf_matrix.shape)
    confusion_annot_df = pd.DataFrame(data=conf_annot_print, index=[f'{label} (True)' for label in labels], columns=[f'{label} (Pred)' for label in labels])

    path = os.path.join(data_folder, dataset_name, model_name, f'{dataset_name}_{model_name}_confusion_matrix.txt')
    with open(path, 'w+') as f:
        print(confusion_annot_df.to_string(), file=f)
    logger.info('Saved the confusion matrix table to %s.', path)

    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.gca()
    mat_ax = ax.matshow(conf_matrix_norm, cmap=sns.color_palette('rocket', as_cmap=True), interpolation='nearest')
    mat_ax.set_clim(vmin=0, vmax=1)
    for spine in ax.spines.values():
        spine.set_edgecolor('#707070')

    if dataset_name.startswith('UOS'):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.1)
        cb = fig.colorbar(mat_ax, cax=cax)
        cb.outline.set_color('#707070')
        cb.ax.tick_params('y', length=6, width=2, which='major', labelsize=20)
        cb.ax.tick_params('y', left=None, right=None, which='minor')
        cb.ax.yaxis.set_ticks(np.arange(0.1, 1.1, 0.1))
        cb.ax.set_ylabel('Count Normalized along True Class', fontsize=20)

    xaxis = np.arange(len(labels))
    #ax.xaxis.tick_bottom()
    ax.set_xticks(xaxis)
    ax.set_xticklabels(labels, rotation=90)

    ax.set_yticks(xaxis)
    #ax.set_yticklabels(labels)
    ax.set_yticklabels([])
    if dataset_name.startswith('UNSW'):
        ax.set_ylabel('True Class', fontsize=20)
    ax.set_xlabel('Class Prediction', fontsize=20)
    ax.xaxis.set_label_position('top') 
    
    ax.set_xticks(np.arange(.5, len(labels)-1, 1), minor=True)
    ax.set_yticks(np.arange(.5-.001*len(labels), len(labels)-1, 1), minor=True)
    ax.grid(which='minor', color='#707070', alpha=1.0, linestyle='-', linewidth=1+(23-len(labels))*0.077)
    ax.tick_params('x', top=None, bottom=None, which='both', labelsize=20)
    ax.tick_params('y', left=None, right=None, which='both', labelsize=20)

    for (i, j), s in np.ndenumerate(conf_annot):
        v = conf_matrix_norm[i, j]
        ax.text(j, i, s, ha='center', va='center', fontsize=31-len(labels)*0.5, color='white' if v < 0.5 else 'black')

    path = os.path.join(data_folder, dataset_name, model_name, f'{dataset_name}_{model_name}_confusion_matrix.pdf')
    plt.savefig(path, dpi=300)
    logger.info('Saved the confusion matrix plot to %s.', path)

def visualize_shap_values(explanations, labels, dataset_names, data_folder, logger):
    """
    Visualize the shap absolute importance values
    """

    fig = plt.figure(figsize=(2.2*len(explanations), 3), dpi=100)
    fig.subplots_adjust(wspace=1.2)
    ax = plt.gca()
    ax.set_axis_off()

    translate_dataset_names = {
        'UNSW-NB15': 'UNSW-NB15',
        'CIC-IDS2017-improved': 'CIC-IDS2017',
        'CSE-CIC-IDS2018-improved': 'CSE-CIC-IDS2018',
        'UOS-IDS23': 'Web-IDS23',
    }

    for i, dataset_name in enumerate(dataset_names):
        ax = plt.subplot(1, len(dataset_names), i+1)

        benign_ind = np.where(np.array(labels[dataset_name]) == 'Benign')[0][0]

        mean_abs_shap = np.mean(np.abs(explanations[dataset_name][:, :, benign_ind].values), axis=0)
        feature_names = np.array(explanations[dataset_name].feature_names)
        ind = np.argsort(-mean_abs_shap)[:5]

        y = mean_abs_shap[ind]
        x = feature_names[ind]
        x = ['\n'.join(textwrap.wrap(f.replace("_", " ").replace(".", " "), width=12 if f.startswith('Down') else 13)) for f in x]

        ax.barh(x, y, height=0.5, color='#003f5c')

        ax.set_axisbelow(True)
        ax.tick_params('x', top=None, bottom=None, which='minor')
        ax.tick_params('x', top=None, length=4, width=2, which='major', labelsize=12, direction='out')
        ax.tick_params('y', length=4, width=2, which='major', labelsize=12, pad=1)
        ax.tick_params('y', left=None, right=None, which='both')
        ax.set_xlabel('Importance', fontsize=12)
        ax.grid()
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.text(0.475, -0.24, translate_dataset_names[dataset_name], ha='center', va='top', transform=ax.transAxes, fontsize=14)

    path = os.path.join(data_folder, 'adv_shap.pdf')
    plt.savefig(path, dpi=300)
    logger.info('Saved the shap plot to %s.', path)

def visualize_init_success(macro_asr, dataset_names, model_names, max_queries, data_folder, attack_name, logger):
    """
    Visualize ASRs for different number of queries
    """

    path = os.path.join(data_folder, f'{attack_name}_asr_table.txt')
    with open(path, 'w+') as f:
        print(pd.DataFrame(macro_asr).to_string(), max_queries, file=f)
    logger.info('Saved the initialization table to %s.', path)

    model_colors = {
        'DNN': '#4053d3',
        'CNN': '#ddb310',
        'AE': '#b51d14',
        'RF': '#00beff',
        'SVM': '#fb49b0',
        'KNN': '#00b25d',
        'Ensemble': '#000000'
    }

    translate_dataset_names = {
        'UNSW-NB15': 'UNSW-NB15',
        'CIC-IDS2017-improved': 'CIC-IDS2017',
        'CSE-CIC-IDS2018-improved': 'CSE-CIC-IDS2018',
        'UOS-IDS23': 'Web-IDS23',
    }
    translate_model_names = {
        'DNN': 'MLP',
        'CNN': 'CNN',
        'AE': 'AE',
        'RF': 'RF',
        'SVM': 'SVM',
        'KNN': 'KNN',
        'Ensemble': 'Ensemble'
    }

    fig = plt.figure(figsize=(11, 3), dpi=100)
    fig.subplots_adjust(wspace=-0.01)

    fig_ax = plt.gca()
    fig_ax.set_axis_off()

    box = fig_ax.get_position()
    fig_ax.set_position([box.x0, box.y0, box.width, box.height])
    legend_handles = [matplotlib.patches.Patch(facecolor=model_colors[model_name], label=translate_model_names[model_name]) for model_name in model_names]
    fig_ax.legend(handles=legend_handles, fontsize=16, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.935), handlelength=2, borderpad=0.15, labelspacing=0.05, columnspacing=0.5, handletextpad=0.25)

    y_max = max(max(max(a) for a in d.values()) for d in macro_asr.values())
    for i, dataset_name in enumerate(dataset_names):
        ax = plt.subplot(1, len(dataset_names), i+1)

        for model_name in model_names:

            ax.plot(max_queries, macro_asr[dataset_name][model_name], linewidth=3.0, color=model_colors[model_name])

        ax.set_axisbelow(True)
        ax.set_ylim([-y_max*0.075, y_max*1.025])
        ax.tick_params('y', length=6, width=2, left=i==0, right=None, which='major', labelsize=16, pad=1)
        ax.tick_params('y', length=4, width=1, left=i==0, right=None, which='minor')
        ax.tick_params('x', top=None, length=6, width=2, which='major', labelsize=16)
        ax.tick_params('x', top=None, bottom=None, which='minor')
        ax.yaxis.set_ticks(np.arange(0, ax.get_ylim()[1], 0.1))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xscale('log')
        ax.set_xticks(max_queries, labels=max_queries)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        if i < len(dataset_names) - 1:
            ax.spines['right'].set_visible(False)
        if i > 0:
            ax.spines['left'].set_visible(False)
        ax.get_xticklabels()[0].set_ha('left')
        ax.get_xticklabels()[-1].set_ha('right')
        ax.get_xticklabels()[-2].set_ha('right')
        ax.text(0.475, -0.225, translate_dataset_names[dataset_name], ha='center', va='top', transform=ax.transAxes, fontsize=16)
        ax.set_xlabel('Max \# Queries', fontsize=14, labelpad=0)
        ax.grid()
        if i > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('ASR', fontsize=14, rotation=0)
            ax.yaxis.set_label_coords(-0.1, 1.015)
    
    path = os.path.join(data_folder, f'{attack_name}_asr_plot.pdf')
    plt.savefig(path, dpi=300)
    logger.info('Saved the initialization plot to %s.', path)

def visualize_init_success_pert(macro_asr, dataset_names, model_names, perturbation_sizes, data_folder, attack_name, logger):
    """
    Visualize ASRs for different maximum relative perturbation sizes
    """

    path = os.path.join(data_folder, f'{attack_name}_asr_pert_table.txt')
    with open(path, 'w+') as f:
        print(pd.DataFrame(macro_asr).to_string(), perturbation_sizes, file=f)
    logger.info('Saved the initialization perturbation table to %s.', path)

    model_colors = {
        'DNN': '#4053d3',
        'CNN': '#ddb310',
        'AE': '#b51d14',
        'RF': '#00beff',
        'SVM': '#fb49b0',
        'KNN': '#00b25d',
        'Ensemble': '#000000'
    }

    translate_dataset_names = {
        'UNSW-NB15': 'UNSW-NB15',
        'CIC-IDS2017-improved': 'CIC-IDS2017',
        'CSE-CIC-IDS2018-improved': 'CSE-CIC-IDS2018',
        'UOS-IDS23': 'Web-IDS23',
    }
    translate_model_names = {
        'DNN': 'MLP',
        'CNN': 'CNN',
        'AE': 'AE',
        'RF': 'RF',
        'SVM': 'SVM',
        'KNN': 'KNN',
        'Ensemble': 'Ensemble'
    }

    fig = plt.figure(figsize=(11, 3), dpi=100)
    fig.subplots_adjust(wspace=-0.01)

    fig_ax = plt.gca()
    fig_ax.set_axis_off()

    box = fig_ax.get_position()
    fig_ax.set_position([box.x0, box.y0, box.width, box.height])
    legend_handles = [matplotlib.patches.Patch(facecolor=model_colors[model_name], label=translate_model_names[model_name]) for model_name in model_names]
    fig_ax.legend(handles=legend_handles, fontsize=16, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.935), handlelength=2, borderpad=0.15, labelspacing=0.05, columnspacing=0.5, handletextpad=0.25)

    y_max = max(max(max(a) for a in d.values()) for d in macro_asr.values())
    for i, dataset_name in enumerate(dataset_names):
        ax = plt.subplot(1, len(dataset_names), i+1)

        for model_name in model_names:

            ax.plot(perturbation_sizes, macro_asr[dataset_name][model_name], linewidth=3.0, color=model_colors[model_name])

        ax.set_axisbelow(True)
        ax.set_ylim([-y_max*0.075, y_max*1.025])
        ax.tick_params('y', length=6, width=2, left=i==0, right=None, which='major', labelsize=16, pad=1)
        ax.tick_params('y', length=4, width=1, left=i==0, right=None, which='minor')
        ax.tick_params('x', top=None, length=6, width=2, which='major', labelsize=14)
        ax.tick_params('x', top=None, bottom=None, which='minor')
        ax.yaxis.set_ticks(np.arange(0, ax.get_ylim()[1], 0.1))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xticks(perturbation_sizes, labels=perturbation_sizes)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        if i < len(dataset_names) - 1:
            ax.spines['right'].set_visible(False)
        if i > 0:
            ax.spines['left'].set_visible(False)
        ax.get_xticklabels()[0].set_ha('left')
        ax.get_xticklabels()[-1].set_ha('right')
        ax.get_xticklabels()[-2].set_ha('right')
        ax.text(0.475, -0.225, translate_dataset_names[dataset_name], ha='center', va='top', transform=ax.transAxes, fontsize=16)
        ax.set_xlabel('Max Perturbation', fontsize=14, labelpad=0)
        ax.grid()
        if i > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('ASR', fontsize=14, rotation=0)
            ax.yaxis.set_label_coords(-0.1, 1.015)
    
    path = os.path.join(data_folder, f'{attack_name}_asr_pert_plot.pdf')
    plt.savefig(path, dpi=300)
    logger.info('Saved the initialization perturbation plot to %s.', path)

def visualize_class_vulnerability(asr, mape, data_folder, dataset_name, attack_name, logger):
    """
    Visualize averaged class vulnerability
    """

    color_asr = '#003f5c'
    color_mape = '#cc8400'

    class_names = sorted(list(asr))
    class_asr = [asr[class_name] for class_name in class_names]
    class_mape = [mape[class_name] if class_name in mape else 0 for class_name in class_names]

    fig, ax1 = plt.subplots(figsize=(10, 2), dpi=100)

    x = np.arange(len(class_names))
    width = 0.4

    ax1.set_xlabel('Class', fontsize=18)
    ax1.set_ylabel('Attack Success Rate', color=color_asr, fontsize=18, loc='top')
    ax1.bar(x, class_asr, width, color=color_asr)
    ax1.yaxis.set_ticks(np.arange(*ax1.get_ylim(), 0.1 if np.max(class_asr) < 0.5 else 0.2))
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    ax2 = ax1.twinx()

    ax2.set_ylabel('MAPE Distance', color=color_mape, fontsize=18, loc='top')
    ax2.bar(x + width, class_mape, width, color=color_mape)
    ax2.yaxis.set_ticks(np.arange(*ax2.get_ylim(), 0.05 if np.max(class_mape) < 0.25 else 0.1))
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    ax1.set_xticks(x + width / 2, class_names)
    ax1.tick_params('x', top=None, bottom=None, which='minor')
    ax1.tick_params('x', top=None, length=6, width=2, which='major', labelsize=14, labelrotation=90, direction='out')
    ax1.tick_params('y', length=6, width=2, which='major', labelsize=18, labelcolor=color_asr)
    ax1.tick_params('y', length=4, width=1, which='minor')
    ax2.tick_params('y', length=6, width=2, which='major', labelsize=18, labelcolor=color_mape)
    ax2.tick_params('y', length=4, width=1, which='minor')

    path = os.path.join(data_folder, dataset_name, f'{dataset_name}_{attack_name}_class_asr_plot.pdf')
    plt.savefig(path, dpi=300)
    logger.info('Saved the class ASR plot to %s.', path)

def visualize_mape(mape, mape_queries, dataset_names, model_names, attack_names, data_folder, logger):
    """
    Visualize MAPE feature distance
    """

    average_mape = {dataset_name: {attack_name: np.mean([mape[dataset_name][model_name][attack_name] for model_name in model_names]) for attack_name in attack_names} for dataset_name in dataset_names}
    average_queries = {dataset_name: {attack_name: np.mean([mape_queries[dataset_name][model_name][attack_name] for model_name in model_names]) for attack_name in attack_names} for dataset_name in dataset_names}

    path = os.path.join(data_folder, 'mape_avg_table.txt')
    with open(path, 'w+') as f:
        print(pd.DataFrame(average_mape).to_string(), pd.DataFrame(average_queries).to_string(), file=f)
    logger.info('Saved the MAPE average table to %s.', path)

    model_colors = {
        'DNN': '#4053d3',
        'CNN': '#ddb310',
        'AE': '#b51d14',
        'RF': '#00beff',
        'SVM': '#fb49b0',
        'KNN': '#00b25d',
        'Ensemble': '#000000'
    }

    translate_dataset_names = {
        'UNSW-NB15': 'UNSW-NB15',
        'CIC-IDS2017-improved': 'CIC-IDS2017',
        'CSE-CIC-IDS2018-improved': 'CSE-CIC-IDS2018',
        'UOS-IDS23': 'Web-IDS23',
    }
    translate_model_names = {
        'DNN': 'MLP',
        'CNN': 'CNN',
        'AE': 'AE',
        'RF': 'RF',
        'SVM': 'SVM',
        'KNN': 'KNN',
        'Ensemble': 'Ensemble'
    }

    for attack_name in attack_names:
        att_name = attack_name.split('_')[-1]

        fig = plt.figure(figsize=(11, 3), dpi=100)
        fig.subplots_adjust(wspace=0)

        x = np.arange(len(model_names))

        ax = plt.gca()
        ax.set_axis_off()

        y_max = max(max(m[attack_name] for m in d.values()) for d in mape.values())
        for i, dataset_name in enumerate(dataset_names):
            ax = plt.subplot(1, len(dataset_names), i+1)

            y = [mape[dataset_name][model_name][attack_name] for model_name in model_names]
            ax.bar(x, y, color=[model_colors[model_name] for model_name in model_names])
            for x_val, y_val in zip(x, y):
                ax.text(x_val, y_val, f'{y_val*100:.0f}\%', ha='center', va='bottom', fontsize=12)

            ax.set_axisbelow(True)
            ax.set_ylim([0, y_max*1.1])
            ax.set_xlim([-1, len(model_names)])
            ax.set_xticks(x, [translate_model_names[model_name] for model_name in model_names], rotation=35)
            ax.tick_params('x', top=None, bottom=None, which='minor')
            ax.tick_params('x', top=None, length=4, width=2, which='major', labelsize=14, direction='out')
            ax.tick_params('y', length=4, width=2, left=i==0, which='major', labelsize=16, pad=1)
            ax.tick_params('y', length=3, width=1, left=i==0, which='minor')
            ax.tick_params('y', right=None, which='both')
            if i < len(dataset_names) - 1:
                ax.spines['right'].set_visible(False)
            if i > 0:
                ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks(np.arange(0, ax.get_ylim()[1], 0.1))
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            ax.text(0.475, -0.24, translate_dataset_names[dataset_name], ha='center', va='top', transform=ax.transAxes, fontsize=16)
            if i > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel('MAPE Feature Distance', fontsize=16)

        path = os.path.join(data_folder, f'{att_name}_mape_plot.pdf')
        plt.savefig(path, dpi=300)
        logger.info('Saved the MAPE plot to %s.', path)

def visualize_attack_features(samples, samples_adv, unsuc_samples, features, data_folder, dataset_name, model_name, attack_name, logger):
    """
    Visualize the attack features
    """

    if (len(samples.index) < 10):
        logger.error('Not enough samples to produce feature plot, only %d provided.', len(samples.index))
        return

    orig_color = '#003f5c'
    trend_color = '#bc5090'
    adv_color = '#ffa600'

    fig = plt.figure(figsize=(3 * len(features), 6), dpi=100)
    fig.subplots_adjust(wspace=0.25)

    fontsize = 18

    ax = fig.gca()
    box = ax.get_position()
    ax.set_position([box.x0 + box.height * 0.02, box.y0, box.width * 0.9, box.height * 0.98])
    equal_vert = np.array([
            [-0.5, 0.1], [0.5, 0.1], [0.5, 0.3], [-0.5, 0.3], [-0.5, 0.1], 
            [-0.5, -0.1], [0.5, -0.1], [0.5, -0.3], [-0.5, -0.3], [-0.5, -0.1]
        ])
    equal_codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY, Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    equal_marker_style = MarkerStyle(Path(equal_vert, equal_codes))
    def make_legend_trend(legend, orig_handle, xdescent, ydescent, width, height, fontsize): # https://stackoverflow.com/questions/22348229/matplotlib-legend-for-an-arrow/22349717#22349717
        return matplotlib.patches.FancyArrow(0, 0.5 * height, width * 0.8, 0, length_includes_head=True, width=0.1 * height, head_width=height, head_length=height)
    legend_handles = [matplotlib.lines.Line2D([0], [0], marker='o', markerfacecolor=orig_color, color='w', markersize=20, label='Originals'), matplotlib.lines.Line2D([0], [0], marker='o', markerfacecolor=adv_color, color='w', markersize=20, label='Successful Adversarials'), matplotlib.patches.FancyArrow(0, 0, 0, 0, color=trend_color, label='Clusters of Original to Adversarial')]
    ax.legend(handles=legend_handles, fontsize=fontsize, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.99), handler_map={matplotlib.patches.FancyArrow : matplotlib.legend_handler.HandlerPatch(patch_func=make_legend_trend),}, handlelength=2, borderpad=0.15, labelspacing=0.01, columnspacing=0.5, handletextpad=0.2)
    ax.set_axis_off()

    for i, f in enumerate(features):
        ax = plt.subplot(1, len(features), i+1)

        orig = samples.loc[:, f].to_numpy()
        adv = samples_adv.loc[:, f].to_numpy()

        view_share = 0.95
        sorted_data = np.sort(np.concatenate([orig, adv]))
        upper_lim = sorted_data[min(int(np.ceil(view_share * len(sorted_data))), len(sorted_data) - 1)]
        lower_lim = sorted_data[int(np.floor((1 - view_share) * len(sorted_data)))]
        lim_diff = upper_lim - lower_lim

        view_mask = (orig <= upper_lim) & (orig >= lower_lim) & (adv <= upper_lim) & (adv >= lower_lim)
        orig, adv = orig[view_mask], adv[view_mask]

        orig_unsuc = unsuc_samples.loc[(unsuc_samples[f] <= upper_lim) & (unsuc_samples[f] >= lower_lim), f].to_numpy() # optionally plot unsuccessful samples as well
        #sns.stripplot(x=np.zeros(orig_unsuc.shape[0]), y=orig_unsuc, hue=['Unsuccessful Original']*orig_unsuc.shape[0], palette=['black'], size=10, alpha=0.01, jitter=0.1, legend=None, ax=ax, rasterized=True)

        plot_labels = np.array(['Original'] * orig.shape[0] + ['Adversarial'] * orig.shape[0])
        plot_data = np.concatenate([orig, adv])
        sns.stripplot(x=np.zeros(plot_data.shape[0]), y=plot_data, hue=plot_labels, palette=[orig_color, adv_color], dodge=True, size=10, alpha=0.1, jitter=0.1, legend=None, ax=ax, rasterized=True)

        # apply feature-individual cluster analysis
        data = np.column_stack((orig, adv))
        minmax = MinMaxScaler().fit(data)
        hdb = HDBSCAN(min_cluster_size=max(2, int(np.round(len(data)*0.01))), cluster_selection_epsilon=0.075, allow_single_cluster=True, store_centers='medoid').fit(minmax.transform(data))
        centers = minmax.inverse_transform(hdb.medoids_)
        logger.info(f'{f}: {centers}') # cluster medoids

        center_size = np.abs(np.diff(centers, axis=-1).flatten())
        sort_ind = np.argsort(-center_size)
        centers = centers[sort_ind, :]
        center_size = center_size[sort_ind]

        pos = np.zeros((len(center_size), 2))
        for i in range(len(center_size)):
            for j in range(i + 1, len(center_size)):
                a1, b1 = np.sort(centers[i, :])
                a2, b2 = np.sort(centers[j, :])
                if max(a1, a2) - lim_diff * 0.0075 <= min(b1, b2) + lim_diff * 0.0075:
                    pos[i, 0] += 1
                    pos[i, 1] += 1
                    pos[j, 1] += 1
        min_gap = 0.02
        max_gap = 0.07
        space = 0.125
        x = np.array([np.clip(2 * space / pos[i, 1], min_gap, max_gap) * (pos[i, 0] - pos[i, 1] / 2) if pos[i, 1] > 0 else 0 for i in range(len(center_size))])

        equal_trends = np.where(center_size < 1e-10)[0]
        large_trends = np.where(center_size / lim_diff >= 0.1)[0]
        small_trends_plus = np.where((center_size / lim_diff < 0.1) & (center_size > 1e-10) & (centers[:, 1] > centers[:, 0]))[0]
        small_trends_minus = np.where((center_size / lim_diff < 0.1) & (center_size > 1e-10) & (centers[:, 1] < centers[:, 0]))[0]

        ax.scatter(x[equal_trends], np.mean(centers[equal_trends, :], axis=-1), s=250, color=trend_color, marker=equal_marker_style, zorder=5, edgecolors='white', linewidths=1)
        ax.quiver(x[large_trends], centers[large_trends, 0], np.full(len(large_trends), 0), centers[large_trends, 1] - centers[large_trends, 0], angles='xy', scale_units='xy', scale=1, color=trend_color, units='dots', width=3.75, headwidth=4.5, headlength=4.5, headaxislength=4.5, zorder=6, edgecolors='white', linewidths=1)
        ax.scatter(x[small_trends_plus], np.mean(centers[small_trends_plus, :], axis=-1), s=250, color=trend_color, marker='^', zorder=5, edgecolors='white', linewidths=1)
        ax.scatter(x[small_trends_minus], np.mean(centers[small_trends_minus, :], axis=-1), s=250, color=trend_color, marker='v', zorder=5, edgecolors='white', linewidths=1)

        ax = plt.gca()
        ax.set_xlim([-0.31, 0.31])
        ax.set_ylim([lower_lim - lim_diff * 0.035, upper_lim + lim_diff * 0.035])
        ax.xaxis.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)
        ax.tick_params('y', length=6, width=2, which='major', labelsize=fontsize, pad=1)
        ax.tick_params('y', length=4, width=1, which='minor')
        ax.tick_params('y', right=None, which='both')
        def thousand_format_ticks(x, pos):
            if x >= 1000:
                return '{:.1f}'.format(x/1000).rstrip('0').rstrip('.') + 'K'
            else:
                return int(x)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(thousand_format_ticks))
        ax.text(0.45, -0.015, '\n'.join(textwrap.wrap(f.replace("_", " ").replace(".", " "), width=15)), ha='center', va='top', transform=ax.transAxes, fontsize=fontsize, linespacing=1)

    path = os.path.join(data_folder, dataset_name, model_name, attack_name, f'{dataset_name}_{model_name}_{attack_name}_features_plot.pdf')
    plt.savefig(path, dpi=300)
    logger.info('Saved the features plot to %s.', path)

def visualize_computational_performance(comp_data, dataset_names, model_names, data_folder, logger):
    """
    Plot computation time of models
    """

    model_colors = {
        'DNN': '#4053d3',
        'CNN': '#ddb310',
        'AE': '#b51d14',
        'RF': '#00beff',
        'SVM': '#fb49b0',
        'KNN': '#00b25d',
        'Ensemble': '#000000'
    }

    translate_dataset_names = {
        'UNSW-NB15': 'UNSW-NB15',
        'CIC-IDS2017-improved': 'CIC-IDS2017',
        'CSE-CIC-IDS2018-improved': 'CSE-CIC-IDS2018',
        'UOS-IDS23': 'Web-IDS23',
    }
    translate_model_names = {
        'DNN': 'MLP',
        'CNN': 'CNN',
        'AE': 'AE',
        'RF': 'RF',
        'SVM': 'SVM',
        'KNN': 'KNN',
        'Ensemble': 'Ensemble'
    }

    fig = plt.figure(figsize=(11, 3), dpi=100)
    fig.subplots_adjust(wspace=0)

    width=0.45
    x = np.arange(len(model_names))

    fig_ax = plt.gca()
    fig_ax.set_axis_off()

    box = fig_ax.get_position()
    fig_ax.set_position([box.x0, box.y0, box.width, box.height])
    legend_handles = [matplotlib.patches.Patch(facecolor=(1, 1, 0, 0), edgecolor='black', label='Training'), matplotlib.patches.Patch(facecolor=(1, 1, 0, 0), edgecolor='black', hatch='//', label='Test Prediction')]
    fig_ax.legend(handles=legend_handles, fontsize=16, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.945), handlelength=2, borderpad=0.15, labelspacing=0.05, columnspacing=0.5, handletextpad=0.25)

    y_max = max(max(max(m.values()) for m in d.values()) for d in comp_data.values())
    for i, dataset_name in enumerate(dataset_names):
        ax = plt.subplot(1, len(dataset_names), i+1)

        y = [comp_data[dataset_name][model_name]['train'] for model_name in model_names]
        ax.bar(x-width*0.485, y, width=width, color=[model_colors[model_name] for model_name in model_names])

        y = [comp_data[dataset_name][model_name]['test'] for model_name in model_names]
        ax.bar(x+width*0.485, y, width=width, color=[model_colors[model_name] for model_name in model_names], hatch='//', edgecolor='black', linewidth=0)

        ax.set_axisbelow(True)
        ax.set_yscale('log')
        ax.set_ylim([0, y_max*1.1])
        ax.set_xlim([-0.8, len(model_names)-0.2])
        ax.set_xticks(x, [translate_model_names[model_name] for model_name in model_names], rotation=35)
        ax.tick_params('x', top=None, bottom=None, which='minor')
        ax.tick_params('x', top=None, length=4, width=2, which='major', labelsize=14, direction='out')
        ax.tick_params('y', length=4, width=2, left=i==0, which='major', labelsize=16, pad=1)
        ax.tick_params('y', length=3, width=1, left=i==0, which='minor')
        ax.tick_params('y', right=None, which='both')
        if i < len(dataset_names) - 1:
            ax.spines['right'].set_visible(False)
        if i > 0:
            ax.spines['left'].set_visible(False)
        ax.text(0.475, -0.24, translate_dataset_names[dataset_name], ha='center', va='top', transform=ax.transAxes, fontsize=16)
        if i > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Computation Time (Minutes)', fontsize=16)

    path = os.path.join(data_folder, 'computational_req_plot.pdf')
    plt.savefig(path, dpi=300)
    logger.info('Saved the computational requirements plot to %s.', path)
