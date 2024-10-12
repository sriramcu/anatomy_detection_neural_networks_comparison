"""
Module to print metric scores derived from a confusion matrix
"""
import os
from pathlib import Path
import pickle
import pprint
import sys
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

from utils.constants import CONF_CSV_DIR, VAL_DIR


def np_average(arr):
    """
    Finds average of a numpy array

    Args:
        arr (np.ndarray): input array

    Returns:
        float: average of input array on a scale of 0-100
    """
    lst = arr.tolist()
    avg = sum(lst)/len(lst)
    avg = avg * 100
    avg = round(avg, 2)
    return avg


def extract_conf_mat(conf_pickle):
    """
    Extract confusion matrix as a 2D array from a pickle file

    Args:
        conf_pickle (str): absolute path to pickle file storing a confusion matrix

    Returns:
        list: confusion matrix
    """
    f = open(conf_pickle, 'rb')    
    results_dict:dict = pickle.load(f)
    f.close()
    
    print(results_dict)
    conf_mat = []
    
    sortednames = sorted(results_dict.keys(), key=lambda x:x.lower())
    for key in sortednames:
        row_dict = results_dict[key]
        sorted_subkeys = sorted(row_dict.keys(), key=lambda x:x.lower())
        row_values = []
        for subkey in sorted_subkeys:
            row_values.append(row_dict[subkey])
        conf_mat.append(row_values)
    
    return conf_mat
   
        
def compute_conf_mat_scores(conf_mat, conf_pickle=""):
    """
    Calculates confusion matrix scores from the confusion matrix 2d array

    Args:
        conf_mat (list): 2d list of confusion matrix
        conf_pickle (str, optional): pickle file name of conf matrix, used only to name the resulting CSV. Defaults to "".
    """
    
    df_cm = pd.DataFrame(conf_mat, range(6), range(6))
    
    print(df_cm)

    # plt.figure(figsize=(10,7))
    # sn.set(font_scale=1.4) # for label size
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g') # font size
    # plt.show()

    TP = np.diag(conf_mat)
    FP = np.sum(conf_mat, axis=0) - TP
    FN = np.sum(conf_mat, axis=1) - TP
    
    num_classes = 6
    l = sum([len(files) for r, d, files in os.walk(VAL_DIR)])
    TN = []
    for i in range(num_classes):
        temp = np.delete(conf_mat, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)        # delete ith column
        TN.append(sum(sum(temp)))
    
    # l = 10000
    debug_computation = False
    if debug_computation:
        for i in range(num_classes):
            print(TP[i] + FP[i] + FN[i] + TN[i] == l)

    precision = TP/(TP+FP)
    sensitivity = TP/(TP+FN)  # recall
    specificity = TN/(TN+FP)
    
    f1 = (2*precision*sensitivity)/(precision+sensitivity)
    
    accuracy = (TP+TN)/(TP+TN+FP+FN)

    # print(TP, TN, FP, FN)
    # print("Classwise accuracy = ", accuracy)
    
    
    if conf_pickle != "":
        csv_dir = CONF_CSV_DIR
        conf_csv = os.path.join(csv_dir, f"{os.path.basename(conf_pickle).split('.')[0]}.csv")
        
        os.system(f"echo 'Precision = {np_average(precision)}' >> {conf_csv}")
        os.system(f"echo 'Accuracy = {np_average(accuracy)}' >> {conf_csv}")
        os.system(f"echo 'Recall (Sensitivity) = {np_average(sensitivity)}' >> {conf_csv}")
        os.system(f"echo 'Specificity = {np_average(specificity)}' >> {conf_csv}")
        os.system(f"echo 'F1 score = {np_average(f1)}' >> {conf_csv}")
    
    else:
        print("Precision, Accuracy, Sensitivity, Specificity, F1 score")
        print(np.average(precision))    
        print(np.average(accuracy))    
        print(np.average(sensitivity))    
        print(np.average(specificity))    
        print(np.average(f1))    


def main():
    conf_pickle = sys.argv[1]
    conf_mat = extract_conf_mat(conf_pickle)
    compute_conf_mat_scores(conf_mat, conf_pickle=conf_pickle)
    
    
if __name__ == "__main__":
    main()
    