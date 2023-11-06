import numpy as np

def metrics_binary(arr_test, arr_pred):
    arr_test = np.array(arr_test)
    arr_pred = np.array(arr_pred)

    arr_pred = np.round(arr_pred)

    TP = sum((arr_test == 1) & (arr_pred == 1))[0]
    TN = sum((arr_test == 0) & (arr_pred == 0))[0]
    FP = sum((arr_test == 1) & (arr_pred == 0))[0]
    FN = sum((arr_test == 0) & (arr_pred == 1))[0]

    res = {'precision': TP / (TP + FP), 'recall': TP / (TP + FN), 'accuracy': (TP + TN) / (TP + TN + FP + FN)}
    res['f_score'] = 2 * res['precision'] * res['recall'] / (res['precision'] + res['recall'])
    res['TP'], res['TN'], res['FP'], res['FN'] = TP, TN, FP, FN

    return res

def metrics_reggresion(arr_test, arr_pred):
    pass


def accuracy_binary(y_true, y_pred):
    return np.sum(y_true == np.round(y_pred)) / y_true.shape[0]

def accuracy_categorical(y_true, y_pred):
    return np.sum(np.argmax(y_pred, axis=1, keepdims=True) == np.argmax(y_true, axis=1, keepdims=True)) / y_true.shape[0]