# Anatomy Detection in Endoscopy using Deep Learning Techniques 

A project used for a comparative study of deep learning algorithms used for anatomy detection in endoscopy. 
Compares the performance of various neural networks and hyperparameters on an eight class classification problem. 
The custom preprocessing (as explained in this README) is loosely based off of the MAPGI [^1] algorithm 
described in [this research paper](./materials/cogan2019.pdf), which also was a starting point for this project 
for us to improve upon by tweaking the hyperparameters and neural networks to find an optimal model.


## Dataset

The [Hyper Kvasir](https://datasets.simula.no/hyper-kvasir/) dataset is used for this project [^2].

## Brief Overview of the repo structure

* The parameters text file under text_files is to log details of past sessions like hyperparameters and optimizer used.
* `constants.py` in the utils folder stores commonly used and modified hyperparameters and directory names (for 
  dataset, video files, etc.) for convenient, single-location program/system tweaking.
* Pickle files in the `training_metrics_pickle_files` folder store training metrics like history (acc, loss for 
  training and validation), hyperparameters, chosen neural network name and the path to saved checkpoints file
  number of epochs, network name, and the path to the saved checkpoints file. The pickle file name is based on the number 
  of epochs and network name for easy reference.
* `analyse_train_data.py` is used to visualise and analyse training metrics saved into the above pickle file, 
  to generate graphs for epoch wise training/validation loss and accuracy, and interval-wise average validation 
  accuracy. The graphs are saved into the `graphs` folder, under a subfolder whose name is based on the pickle 
  file's name.
* We have chosen to use this pickle file system to save us the trouble of mentioning what network was used, where 
  the checkpoint file is located, whether preprocessing is enabled, which would affect the image operations on 
  the test data. It is also useful in case you want to write a separate program to analyse model training 
  metrics to see how to 
  optimise training further.
* `confusion_matrix_tables` is a folder which stores the CSV files of the confusion matrix generated when we run 
  the evaluate program on the test data.

## Usage

Follow these steps in order after cloning the repo:

### 1. Setup
`pip install -r requirements.txt`  

### 2. Download Dataset
The relevant eight classes have already been extracted from the Hyper Kvasir dataset and uploaded to Google Drive.
[Download the dataset](https://drive.google.com/file/d/1yFgW7AAySYK7Dkj2VWO4fiyIYpVA-ym2/view?usp=sharing) and 
place it in the `dataset` folder after unzipping it. After doing this, the directory structure would look like:

```
This Project
├───dataset
│   ├───train
│   │   ├───cecum
│   │   │    ├────── (...).jpg
│   │   │ 
│   │   ├───dyed-lifted-polyps
│   │   │    ├────── (...).jpg
│   │   │ 
│   │   ├───dyed-resection-margins
│   │   │    ├────── (...).jpg
(..............)
```


### 3. Split Dataset

Split dataset into train, validation and test datasets using `split_dataset.py` script. Default split is 0.7 - 
0.15 - 0.15. Use the following command to change that, for example, to a 60-20-20 ratio instead:

`python split_dataset.py -d dataset/train -v 0.2 -t 0.2`  

To reverse the above split, use `python merge_datasets.py`

### 4. Training

We can modify image augmentation techniques in utils/my_datagen.py. Or we could change L2 parameter, initial 
learning rate or dropout rate of final dense (fully connected) layer in utils/constants.py. Then, choose your 
neural network via the command line. We suggest using EfficientNetB7.

`python train.py -e <num_epochs> -n efficientnet -c 8 -p 1`  

Where e is for number of epochs, n is for neural network name, c is for number of classes (should be set to 
number of classes in constants.py `CLASS_LABELS` which is equal to number of subfolders in train dataset), 
and p is to select whether custom preprocessing is enabled for training (if so, it will automatically be 
enabled for prediction by storing this flag in the pickle file as explained). 

* Note that we do not need to specify dataset directories since these are given by default in utils/constants.
  py. `-t` and `-v` arguments are optional to change these.
* Try to make sure that in each training session you use a different number of epochs since pickle file name, 
  checkpoint file name, etc. are based on number of epochs, and they could get overwritten if all are the same.
* To avoid the above case, the `-u` argument can be used to append a custom string to the checkpoint file name.
* `neural_networks.py` contains all the neural networks used in this project (specified by `-n` argument).

### 5. Analyze Training Metrics

* After training, a pickle file will be created in `training_metrics_pickle_files/` which contains data from 2 
  pickle.dump() statements - the first dumps the training "history" which is a dictionary where the keys are 
  'acc', 'loss', 'val_acc', 'val_loss'. The second dumps a list containing the number of epochs, 
  hyperparameters, network name and the path to saved checkpoints file (i.e. the list saved to 
  text_files/parameters.txt with the checkpoints filepath appended to it).

`python analyse_train_data.py -p training_metrics_pickle_files/train_metrics_<num_epochs>_<network>.pickle`  

`-k` and `-i` are optional arguments for kth highest accuracy and interval-wise average accuracy respectively.  

This will generate graphs for epoch-wise training/validation loss and accuracy:

![accuracy_graph.jpg](graphs%2Ftrain_metrics_307epochs_efficientnet_eight_class%2Faccuracy_graph.jpg)


### 6. Evaluation

`python evaluate.py -p training_metrics_pickle_files/train_metrics_<num_epochs>_<network>.pickle`

This will run model predictions on the test dataset directory and generate a confusion matrix for the 
trained model in the form of a csv file complete with confusion matrix-related metrics (like accuracy, F1 score,
precision, recall, etc.) under the "confusion_matrix_tables" directory.  

`-d` is an optional argument to specify a different test dataset directory.

### 7. Dual Annotation
We carry out steps 1 to 4 until we are satisfied with validation accuracy as well as the training metrics. Then,
use the best model in the dual annotation program (performing expert-based and AI-based predictions), for which 
the command is shown below:

`python evaluate_video.py -i videos/input_videos/input.mkv -o videos/output_videos/output.mkv -t 
training_metrics_pickle_files/train_metrics_<num_epochs>_<network>.pickle`

* The program, after performing frame wise model predictions on the mkv video, then prompts the user for "expert 
  annotations", which are entered as inputs via the command line, to simulate actual medical experts laying out 
  the ground truth for the video.
* The input video is present in the repository in the path mentioned in the above command. We created this 
  video by stitching together labelled videos in the Hyper Kvasir dataset. It is a video showing performed 
  endoscopy, in regions of the body covered by the eight classes.
* If the above video is being used, the expert annotations are: 0-60 seconds are z-line, 60-92 seconds are 
  cecum and 92 seconds till the end of the video are polyp.
* Note that only mkv files are supported by the system at present since we hardcoded the fourcc code of the codec 
used in the OpenCV video writer functions. 
* The output video is annotated in the top left corner in yellow color with expert-entered labels (via the 
  command line in the form of user input prompted by the program) as well as in the top right corner in green 
  color with the AI-based prediction returned by the model in the form of the predicted class label and the prediction probability. 

## Note

Due to large file restrictions, the checkpoints have not been uploaded to the repository, and they remain in our 
local system. Some checkpoint files have been uploaded to 
[this drive folder](https://drive.google.com/drive/folders/1obA05irsQN6eW3pqgrjSUXsSZNkav6yD?usp=sharing). 
Please reach out to me if you'd like to use a specific model checkpoints (h5) file for 
predictions, will be happy to share it. You can also contact me for any other kind of clarification.

## Analysis

The analysis of results obtained using different configurations such as neural network used, choice of 
augmentations and custom preprocessing can be found in `analysis/analysis.xlsx`. This Excel sheet shows the 
training and validation loss graphs obtained as well as confusion matrix metrics in each case. In multiple such 
configurations, we were able to surpass the accuracy obtained by the Cogan [^1] paper. 

## Contributions

Contributions are welcome! If you have any ideas, bug fixes, GUI enhancements or anything else, please feel 
free to open an issue or submit a pull request.

## References

[^1]: Cogan T, Cogan M, Tamil L. MAPGI: Accurate identification of anatomical landmarks and diseased tissue in 
gastrointestinal tract using deep learning. Comput Biol Med. 2019 Aug;111:103351. doi: 10.1016/j.compbiomed.
2019.103351. Epub 2019 Jul 10. PMID: 31325742.

[^2]: Borgli, H., Thambawita, V., Smedsrud, P. H., Hicks, S., Jha, D., Eskeland, S. L., Randel, K. R., 
Pogorelov, K., Lux, M., Nguyen, D. T. D., Johansen, D., Griwodz, C., Stensland, H. K., Garcia-Ceja, E., Schmidt,
P. T., Hammer, H. L., Riegler, M. A., Halvorsen, P., & De Lange, T. (2020). HyperKvasir, a comprehensive 
multi-class image and video dataset for gastrointestinal endoscopy. Scientific Data, 7(1).
https://doi.org/10.1038/s41597-020-00622-y