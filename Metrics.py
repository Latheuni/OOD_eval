## packages
import numpy as np
from sklearn import metrics



def acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc


# Code from openOOD benchmark:  https://github.com/Jingkang50/OpenOOD/blob/main/openood/evaluators/metrics.py#L40
# accessed 05/03/2024
def auc_and_fpr_recall(conf, label, tpr_th):
    """
    Evaluates if the method correctly identified the OOD samples by assigning low confidence scores
    Label should be an indicator: OOD or not (in real-life)

    """
    # following convention in ML we treat OOD as positive
    print('label', label)
    ood_indicator = np.zeros_like(label)
    print('OOD_indicator', ood_indicator)
    ood_indicator[label == -1] = 1
    print(conf)
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
