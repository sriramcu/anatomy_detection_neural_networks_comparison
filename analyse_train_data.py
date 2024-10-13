"""
Module to analyse training metrics from the train history stored in a pickle file generated after training
Meant to understand how the validation accuracy changes over epochs, to find out which models gradually increased val acc
"""
import argparse
import os
import pickle

from matplotlib import pyplot as plt

from utils.constants import GRAPHS_ROOT_DIR


def list_average(a: list, rnd: bool = True):
    """
    Function that returns the rounded average of a list on a scale of 0 to 100

    Args:
        a (list): input list
        rnd (bool, optional): controls whether returned average is rounded off. Defaults to True.

    Returns:
        float: average of the list on a scale of 0 to 100
    """
    avg = sum(a) / len(a)
    avg = avg * 100
    if rnd:
        avg = round(avg, 1)
    return avg


def split_list(input_list, num_sublists):
    """
    Splits a list of numbers to a list of n equally sized sublists
    If a precisely equal split is not possible, then the first few sublists will have one extra element each.
    Meant to show rise of train/val accuracy/loss periodically (eg-after every 10 epochs)

    Args:
        input_list (list): list of float values
        num_sublists (int): number of intervals to split the list into

    Returns:
        list: list of n sublists
    """
    k, m = divmod(len(input_list), num_sublists)
    return [input_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_sublists)]


def kth_highest_acc(k: int, lst: list, rnd: bool = True) -> float:
    """
    The kth_highest_acc function takes a list of accuracies and returns the kth highest accuracy.
    If there are less than 2 accuracies in the list, then it will return 0.
    
    Args:
        k(int): Specify the kth highest accuracy
        lst(list): Specify the list of accuracies that we want to find the kth highest accuracy from
        rnd(bool)=True: Specify whether to round the result to one decimal place
    
    Returns:
        bool: The kth highest accuracy in a list

    """

    k = min(k, len(lst) - 2)
    k = max(0, k)
    if k == len(lst) - 1 and k == 0:
        k = -1
    res = sorted(lst, reverse=True)[k + 1]
    res = res * 100
    if rnd:
        res = round(res, 1)
    return res


def analyse_train_data(pickle_file_path, intervals, k):
    """
    Driver function that prints the average, interval wise and kth highest validation accuracy\
        from the train history stored in the pickle file specified as a function argument

    Args:
        pickle_file_path (str): filepath to train pickle file
        intervals (int): number of intervals to split the validation accuracy list into
        k (int): Specify k to find out the kth highest accuracy
    """

    f = open(pickle_file_path, 'rb')
    hist = pickle.load(f)
    f.close()
    val_acc_list = hist['val_accuracy']
    print(f"Average val accuracy = {list_average(val_acc_list)}")
    interval_wise_val_accs = [list_average(sublist)
                              for sublist in split_list(val_acc_list, intervals) if sublist != []]
    print(f"Interval Wise Val Acc = {interval_wise_val_accs}")
    l = len(val_acc_list)
    print(f"Peak val accuracy (ranked {k} out of {l} epochs) = {kth_highest_acc(k, val_acc_list)}")


def visualise_from_pickle(pickle_file_path):
    """
    Loads train metadata into a dictionary from a pickle file and calls the visualise function on this metadata

    Args:
        pickle_file_path (str): absolute path of the pickle file generated after training

    Raises:
        ValueError: if the pickle file has more than one dot (.) in its file name
    """
    graph_folder_name = pickle_file_path.split(".")  # so that graph folder name has model name and number of epochs
    if len(graph_folder_name) > 2:
        raise ValueError(f"Too many dots ({len(graph_folder_name) - 1}) in pickle file path (max 1)")
    graph_folder_name = graph_folder_name[0]

    f = open(pickle_file_path, 'rb')
    hist = pickle.load(f)
    f.close()

    visualise(hist, graph_folder_name)


def visualise(hist: dict, graph_folder_name):
    """
    Visualises and saves graphs generated on a dictionary containing validation & training loss & accuracy

    Args:
        hist (dict): dictionary containing validation & training loss & accuracy values
        graph_folder_name (str, optional): base name of folder in which graph img is stored. Defaults to None.
    """
    graph_folder_path = os.path.join(GRAPHS_ROOT_DIR, os.path.basename(graph_folder_name))
    if os.path.exists(graph_folder_path):
        i = 1
        while os.path.exists(graph_folder_path + str(i)):
            i += 1
        graph_folder_path += str(i)
    os.makedirs(graph_folder_path)

    plt.plot(hist['accuracy'])
    plt.plot(hist['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    graph_file_name = "accuracy_graph.jpg"
    graph_file_path = os.path.join(graph_folder_path, graph_file_name)
    plt.savefig(graph_file_path)
    plt.show()

    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    graph_file_name = "loss_graph.jpg"
    graph_file_path = os.path.join(graph_folder_path, graph_file_name)
    plt.savefig(graph_file_path)
    plt.show()


def main():
    """
    Main function that calls the driver function based on user provided pickle file path, k value, \
        and the number of intervals
    """
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k", help="return kth largest val acc obtained", type=int, default=2)
    parser.add_argument("-i", "--intervals", help="number of time intervals to display for val accuracy",
                        type=int, default=5)
    parser.add_argument("-p", "--train_pickle_file", help="Training pickle file to analyse",
                        type=str, required=True)
    args = vars(parser.parse_args())

    pickle_file_path = os.path.abspath(args["train_pickle_file"])
    intervals = int(args["intervals"])
    k = int(args["k"])

    analyse_train_data(pickle_file_path, intervals, k)
    visualise_from_pickle(pickle_file_path)


if __name__ == "__main__":
    main()
