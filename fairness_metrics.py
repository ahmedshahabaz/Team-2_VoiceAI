
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

def chi_DIR_plot(dataset, _opensmile_df_, ground_truth, _predictions_, attribute='gender', calc_chi_square=True):
    
    #print("---" , attribute.upper(), "---")
    
    sensitive_attribute = _opensmile_df_[attribute]
    if attribute == 'age':
        sensitive_attribute = _opensmile_df_['AGE_bin']
        
    contingency_table = pd.crosstab(sensitive_attribute, _predictions_, rownames=[attribute], colnames=['Prediction'])
    
    # Compare distributions
    _print_string_ = f'---{attribute.upper()}---\n'
    if calc_chi_square:
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        chi_square_result = {
            "chi2_stat": chi2_stat,
            "p_value": p_value,
            "dof": dof,
            "expected": expected
        }
        #print(f'Chi-Square Statistic: {chi2_stat}, p-value: {p_value}')
        _print_string_ += f'Chi-Square Statistic: {chi2_stat}, p-value: {p_value}\n'

    unique_groups = sensitive_attribute.unique()
    favorable_outcome = {}

    def age_map(x):
        labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
        return labels[x]

    if attribute == 'gender':
        map_func = dataset.map_gender_back
    elif attribute == 'site':
        map_func = dataset.map_site_back
    elif attribute == 'age':
        map_func = age_map
        
    accuracies = [_predictions_ == ground_truth]
    for group in unique_groups:
        #group_predictions = _predictions_[sensitive_attribute == group]
        group_predictions = accuracies[0][sensitive_attribute == group]
        group_truth = ground_truth[sensitive_attribute == group]
        
        favorable_outcome[group] = np.mean(group_predictions)
        
    most_favorable_group = max(favorable_outcome, key=favorable_outcome.get)
    disparate_impact_ratios = {group: favorable_outcome[group] / favorable_outcome[most_favorable_group] for group in unique_groups}
    
    #Print the disparate impact ratios
    for group, ratio in disparate_impact_ratios.items():
        _print_string_ += f'Disparate Impact Ratio for Group {map_func(group)}: {ratio}\n'
        #print(f'Disparate Impact Ratio for Group {map_func(group)}: {ratio}')

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