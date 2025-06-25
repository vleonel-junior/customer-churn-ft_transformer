from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score, balanced_accuracy_score
import numpy as np

def performance(labels, probs, thresold=0.5, name = 'test_dataset', path_save = None):
    #--------------------------------------------
    if thresold != 0.5:
        predicted_labels = []
        for prob in probs:
            if prob >= thresold:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
    else:
        predicted_labels = np.round(probs)

    #---------------------------------------------
    tn, fp, fn, tp  = confusion_matrix(labels, predicted_labels).ravel()
    acc             = np.round(accuracy_score(labels, predicted_labels), 4)
    ba              = np.round(balanced_accuracy_score(labels, predicted_labels), 4)
    roc_auc         = np.round(roc_auc_score(labels, probs),4)
    pr_auc          = np.round(average_precision_score(labels, probs), 4)
    mcc             = np.round(matthews_corrcoef(labels, predicted_labels), 4)
    sensitivity     = np.round(tp / (tp + fn), 4)
    specificity     = np.round(tn / (tn + fp), 4)
    precision       = np.round(tp / (tp + fp), 4)
    f1              = np.round(2*precision*sensitivity / (precision + sensitivity), 4)
    ck              = np.round(cohen_kappa_score(labels, predicted_labels), 4)
    print('Performance for {}'.format(name))
    print('AUC-ROC: {}, AUC-PR: {}, Accuracy: {}, B_ACC : {}, MCC: {}, Sensitivity/Recall: {}, Specificity: {}, Precision: {}, F1-score: {}, CK-score {}'.format(
          roc_auc, pr_auc, acc, ba, mcc, sensitivity, specificity, precision, f1, ck))
    result = {}
    result['Dataset'] = name
    result['AUC'] = roc_auc
    result['PR_AUC'] = pr_auc
    result['ACC'] = acc
    result['B_ACC'] = ba
    result['MCC'] = mcc
    result['Sensitivity'] = sensitivity
    result['F1'] = f1
    result['Pre'] = precision
    result['Spe'] = specificity
    result['Ck']  = ck
    
    return roc_auc, pr_auc, acc, ba, mcc, sensitivity, specificity, precision, f1, ck