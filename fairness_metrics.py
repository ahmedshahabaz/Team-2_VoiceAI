
# https://towardsdatascience.com/mitigating-bias-in-ai-with-aif360-b4305d1f88a9
# https://aif360.readthedocs.io/en/stable/modules/generated/aif360.sklearn.metrics.disparate_impact_ratio.html

from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from fairness import tools, balancers


def chi_DIR_plot(audio_dataset, _opensmile_df_, ground_truth, _predictions_, attribute='gender', calc_chi_square=True):
    
    print("---" , attribute.upper(), "---")
    _print_string_ = ''
    sensitive_attribute = _opensmile_df_[attribute]
    if attribute == 'age':
        sensitive_attribute = _opensmile_df_['AGE_bin']
        
    # accuracies = [_predictions_ == ground_truth]
    accuracies = np.array(_predictions_) == np.array(ground_truth)
    
    unique_groups = sensitive_attribute.unique()
    favorable_outcome = {}

    def age_map(x):
        labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
        return labels[x]

    if attribute == 'gender':
        map_func = audio_dataset.map_gender_back
    elif attribute == 'site':
        map_func = audio_dataset.map_site_back
    elif attribute == 'age':
        map_func = age_map
    
    tpr_fpr_data = {}
    
    for group in unique_groups:
        group_indices = sensitive_attribute == group
        group_predictions = _predictions_[group_indices]
        group_truth = ground_truth[group_indices]

        tp = np.sum((group_predictions == 1) & (group_truth == 1))
        fp = np.sum((group_predictions == 1) & (group_truth == 0))
        fn = np.sum((group_predictions == 0) & (group_truth == 1))
        tn = np.sum((group_predictions == 0) & (group_truth == 0))

        tpr = tp / (tp + fn)  # True Positive Rate
        fpr = fp / (fp + tn)  # False Positive Rate

        tpr_fpr_data[group] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'TPR': tpr,
            'FPR': fpr
        }
        
        favorable_outcome[group] = np.mean(group_predictions == group_truth)
    
    most_favorable_group = max(favorable_outcome, key=favorable_outcome.get)
    disparate_impact_ratios = {group: favorable_outcome[group] / favorable_outcome[most_favorable_group] for group in unique_groups}

    contingency_table = pd.DataFrame(tpr_fpr_data).T
    contingency_table.index = [map_func(group) for group in contingency_table.index]

    print(contingency_table)

    if calc_chi_square:
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        chi_square_result = {
            "chi2_stat": chi2_stat,
            "p_value": p_value,
            "dof": dof,
            "expected": expected
        }
        _print_string_ += f'Chi-Square Statistic: {chi2_stat}, p-value: {p_value}\n'
        
        if p_value <= 0.05:
            _print_string_ += 'Dependent\n'
        else:
            _print_string_ += 'Independent\n'
    
    # Print the disparate impact ratios
    for group, ratio in disparate_impact_ratios.items():
        _print_string_ += f'Disparate Impact Ratio for Group {map_func(group)}: {ratio}\n'

    # Plotting the disparate impact ratios
    plt.figure(figsize=(10, 6))
    groups = [map_func(group) for group in unique_groups]
    ratios = [disparate_impact_ratios[group] for group in unique_groups]
    plt.bar(groups, ratios, color=plt.get_cmap('tab10').colors[:len(unique_groups)])
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Rule Threshold')
    plt.xlabel('Groups')
    plt.ylabel('Disparate Impact Ratio')
    plt.title('Disparate Impact Ratios by Group')
    plt.legend()
    plt.show()

    return chi_square_result, disparate_impact_ratios, _print_string_


# Scott Lee, “scotthlee/fairness: First release!”. Zenodo, Jun. 01, 2021. doi: 10.5281/zenodo.4890946.
# https://github.com/scotthlee/fairness/tree/master?tab=readme-ov-file

# https://developers.google.com/machine-learning/glossary/fairness#s
# https://pubs.rsna.org/page/ai/blog/2023/08/ryai_editorsblog082523

# https://github.com/gpleiss/equalized_odds_and_calibration

def equalized_metrics(_opensmile_df_, y_gt, y_pred, attribute='gender'):

    print("---" , attribute.upper(), "---")
    sensitive_attribute = _opensmile_df_[attribute]
    pred_stats = tools.clf_metrics(y_gt, y_pred)
    print(pred_stats)
    
    pb = balancers.BinaryBalancer(y=y_gt, y_=y_pred, a=sensitive_attribute, summary=False)
    print("\nEqualized ODDs")
    pb.adjust(goal='odds', summary=False)
    pb.summary()
    #pb.plot(xlim=(0, 0), ylim=(0, 0), lp_lines=False, roc_curves=False)

    print("Equal Opportunity")
    pb.adjust(goal='opportunity', summary=False)
    pb.summary()
    #pb.plot(xlim=(0, 0), ylim=(0, 0), lp_lines=False, roc_curves=False)