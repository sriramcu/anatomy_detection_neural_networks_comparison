"""
Module to show/save training graphs
"""
import sys
import matplotlib.pyplot as plt
import pickle
import os

from constants import GRAPHS_ROOT_DIR
    
        
def main():
    """
    Main function calls visualise_from_pickle function where pickle file path is the only cmd line argument
    """   
    if len(sys.argv) < 2:
        print(f"Usage: python3 {__file__} <train_metrics_pickle_file>")
        sys.exit(-1)
        
        
    pickle_file_path = os.path.abspath(sys.argv[1])
    visualise_from_pickle(pickle_file_path)     


def visualise_from_pickle(pickle_file_path):
    """
    Loads train metadata into a dictionary from a pickle file and calls the visualise function on this metadata

    Args:
        pickle_file_path (str): absolute path of the pickle file generated after training

    Raises:
        ValueError: if the pickle file has more than one dot (.) in its file name
    """
    graph_folder_name = pickle_file_path.split(".")
        
    if len(graph_folder_name) > 2:
        raise ValueError(f"Too many dots ({len(graph_folder_name)-1}) in pickle file path (max 1)")
    
    graph_folder_name = graph_folder_name[0]
    
    f = open(pickle_file_path, 'rb')
    hist = pickle.load(f)
    f.close()

    visualise(hist, graph_folder_name=graph_folder_name)
    
    
def visualise(hist: dict, graph_folder_name=None):   
    """
    Visualises and saves graphs generated on a dictionary containing validation & training loss & accuracy

    Args:
        hist (dict): dictionary containing validation & training loss & accuracy values
        graph_folder_name (str, optional): base name of folder in which graph img is stored. Defaults to None.
    """    
    
    plt.plot(hist['accuracy'])
    plt.plot(hist['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    if graph_folder_name:
        graph_folder_path = os.path.join(GRAPHS_ROOT_DIR, os.path.basename(graph_folder_name))
        
        os.makedirs(graph_folder_path, exist_ok=True)
        graph_file_name = f"accuracy_graph.jpg"
        graph_file_path = os.path.join(graph_folder_path, graph_file_name)
        plt.savefig(graph_file_path)
    
    plt.show()     
    
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    if graph_folder_name:
        graph_file_name = f"loss_graph.jpg"
        graph_file_path = os.path.join(graph_folder_path, graph_file_name)
        plt.savefig(graph_file_path)
        
    plt.show()        
        
        

if __name__ == "__main__":
    
    main()
    
        