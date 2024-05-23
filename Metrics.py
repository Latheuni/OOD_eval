## packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

# Code from openOOD benchmark:  https://github.com/Jingkang50/OpenOOD/blob/main/openood/evaluators/metrics.py#L40
# accessed 05/03/2024
def convert_ind(ind_array):
    """
    convert -1, 1 to corresponding True/False boolean araay (for OOD indicator)
    """
    return(np.array([True if i == -1 else False for i in ind_array]))

def general_metrics(conf, OOD_ind, predictions, ytrue, verbose):
    """
    Calculated ID and OOD accuracy score and balanced accuracy score
    """
    if not isinstance(OOD_ind, np.ndarray):
        OOD_ind = np.array(OOD_ind)
        if verbose:
            print('Converted OOD_ind', OOD_ind)

    OOD_pred = pd.DataFrame(predictions).iloc[convert_ind(OOD_ind),:]
    ID_pred = pd.DataFrame(predictions).iloc[~convert_ind( OOD_ind),:]
    OOD_conf = pd.DataFrame(conf).iloc[convert_ind(OOD_ind),:]
    ID_conf = pd.DataFrame(conf).iloc[~convert_ind( OOD_ind),:]
    OOD_ytrue = pd.DataFrame(ytrue).iloc[convert_ind(OOD_ind),:]
    ID_ytrue = pd.DataFrame(ytrue).iloc[~convert_ind( OOD_ind),:]

    return metrics.accuracy_score(OOD_ytrue, OOD_pred),metrics.accuracy_score(ID_ytrue, ID_pred), metrics.balanced_accuracy_score(OOD_ytrue, OOD_pred), metrics.balanced_accuracy_score(ID_ytrue, ID_pred)

def Accuracy_reject_curves(conf_scores, ytrue, predictions):
    """
    Calculate AR curves (<= threshold is rejected)
    """
    step = 0.01
    steps = np.arange(0, 1, step)
    acc = np.zeros(len(steps))
    perc = np.zeros(len(steps))
    # Loop over steps
    for i in range(0, len(steps)):
        t = steps[i]
        ind = [bool(s > t) for s in conf_scores]
        perc[i] = (len(ind) - sum(ind))/len(ind)
        if sum(ind) > 0:
            acc[i] = metrics.accuracy_score(pd.DataFrame(ytrue).iloc[ind], pd.DataFrame(predictions).iloc[ind])
        else:
            acc[i] = 1.0
    results = pd.DataFrame([acc, steps, perc]).T
    results.columns = ['Acc', 'Steps', "Percentage"]
    return results

def plot_AR_curves(results, file_dir):
    """
    Plot AR curve
    """
    x = sns.lineplot(results, x = "Steps", y = "Acc")
    fig = x.get_figure()
    fig.savefig(file_dir)

def auc_and_fpr_recall(conf, label, tpr_th):
    """
    Evaluates if the method correctly identified the OOD samples by assigning low confidence scores
    Label should be an indicator: OOD or not (in real-life)
    """
    # following convention in ML we treat OOD as positive
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in = metrics.precision_recall_curve(
        1 - ood_indicator, conf
    )

    precision_out, recall_out, thresholds_out = metrics.precision_recall_curve(
        ood_indicator, -conf
    )

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr
