# Anatomy Detection using Deep Learning Techniques 

An project used for Anatomy Detection Using Deep Learning. Compares the performance of various neural networks 
and hyperparameters on a six class classification problem. The custom preprocessing (as explained in this 
README) is loosely based off of the MAPI algorithm, [Timothy Cogan et al](./materials/cogan2019.pdf).   

The methodology and results of this work is described [in this powerpoint presentation](./materials/AnatomyDetection_Inceptionv4_cogan.pptx)
The ppt shows the effect of using different neural networks, optimizers, augmentation techniques as well as the 
usage of the custom preprocessing technique.

## Dataset

The [Hyper Kvasir](https://datasets.simula.no/hyper-kvasir/) dataset is used for this project [^1].

## Brief Overview of the repo structure

* All programs, functions and modules have been documented with appropriate docstrings. This includes a 
  summarised explanation of each program's purpose at the beginnning of the code. It is advised to go through 
  these before using the code to get an idea of how each program or function interact with the others.

* All standalone executable programs, i.e., the python programs in the root directory of this project, show the command line usage by running `python3 <name_of_the_program> -h`.  

* The text files are for reference, and to remember details of past sessions like hyperparameters used, terminal outputs, etc.

* The shell scripts are meant to be used if we want to do some tasks overnight sequentially without any user intervention.

* `constants.py` in the utils folder stores commonly used and modified hyperparameters and directory names (for dataset, video files, etc.) for convenient, single-location program/system tweaking.



## Usage

Follow these steps in order:

### 1. Setup
`pip install -r requirements.txt`  

### 2. Download Dataset
The relevant six classes have already been extracted from the hyper kvasir dataset and segregated into training and validation dataset (in an 80-20 ratio) and uploaded to Google Drive. It can be downloaded by running the following command in the root directory of this Anatomy Detection Project:-
`./shell_scripts/download_dataset.sh`

### 3. Training
We can modify image augmentation techniques in utils/my_datagen.py. Or we could change L2 parameter, initial learning rate or dropout rate of final dense (fully connected) layer in utils/constants.py. Then, choose your neural network. We suggest using EfficientNetB7, since it was the best model as discussed in the ppt linked to at the beginning of this README.  

`python3 train.py -e <num_epochs> -n efficientnet -c 6 -p 1`  

Where e is for number of epochs, n is for neural network name, c is for number of classes and p is to select whether custom preprocessing is enabled for training (if so, it will automatically be enabled for prediction by storing this flag in the pickle file as explained below). Note that we do not need to specify dataset directories since these are given in utils/constants.py. Try to make sure that in each training session you use a different number of epochs since pickle file name, checkpoint file name, etc. are based on number of epochs.
After training, a pickle file will be created in `training_metrics_pickle_files/` which contains data from 2 pickle.dump() statements- the first dumps the training "history" which is a dictionary where the keys are 'acc', 'loss', 'val_acc', 'val_loss'. The second dumps a list containing the number of epochs, hyperparameters, network name and the path to saved checkpoints file (i.e. the list saved to text_files/parameters.txt with the checkpoints filepath appended to it).   

To observe graphs for epoch wise training/validation loss and accuracy, run the following command:
`python3 utils/visualise_train_data.py <train_metrics_pickle_file>`  

To analyse interval-wise average validation accuracy as well as peak validation accuracy (ranked kth from the best validation accuracy over the entire training session), run the following command:
`python3 utils/analyse_train_data.py -pfile <train_metrics_pickle_file>`



### 4. Evaluation
Note the file path of the pickle file generated in step 3. Then, run the following command:
`python3 evaluate.py -pfile <train_pickle_file_path>`

This will run model predictions on the validation dataset directory and generate a confusion matrix for the trained model in the form of a pickle file, stored under the "confusion_matrix_pickle_files" directory as well as a csv file (that can be viewed as a table in any spreadsheet software like Excel, Google Sheets or LibreOffice Calc) complete with confusion matrix-related metrics (like accuracy, F1 score, precision, recall, etc.) under the "confusion_matrix_tables" directory.  


### 5. Dual Annotation
We carry out steps 1 to 4 until we are satisfied with validation accuracy as well as the metrics discussed in step 4. Then, we note the pickle file path for the best performing trained model and use it in the dual annotation program (performing expert-based and AI-based predictions), for which the command is shown below:

`python3 main.py -i videos/input_videos/<input_video_name> -o videos/output_videos/<output_video_name> -tp <train_pickle_path_for_best_performing_model>` 

Note that only mkv files are supported by the system at present since we hardcoded the fourcc code of the codec used in the OpenCV video writer functions.
The input video is a video showing performed endoscopy, in regions of the body covered by the six classes noted in the ppt. The output video is annotated on the top left corner in yellow color with expert-entered labels (via the command line in the form of user input prompted by the program) as well as on the top right corner in green color with the AI-based prediction returned by the model in the form of the predicted class label and the prediction probability. 


## References

[^1]: Borgli, H., Thambawita, V., Smedsrud, P. H., Hicks, S., Jha, D., Eskeland, S. L., Randel, K. R., 
Pogorelov, K., Lux, M., Nguyen, D. T. D., Johansen, D., Griwodz, C., Stensland, H. K., Garcia-Ceja, E., Schmidt,
P. T., Hammer, H. L., Riegler, M. A., Halvorsen, P., & De Lange, T. (2020). HyperKvasir, a comprehensive 
multi-class image and video dataset for gastrointestinal endoscopy. Scientific Data, 7(1).
https://doi.org/10.1038/s41597-020-00622-y