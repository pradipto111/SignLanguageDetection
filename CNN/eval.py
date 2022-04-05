from utils import *
from model import ASLResnet, ASLMobilenet
import numpy as np
from matplotlib import pyplot as plt
import torch, os, glob
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.models as models
import torchvision.transforms as tt
import cv2, argparse, os
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", type=str, help = "Path to the dataset")
parser.add_argument("model", type=str, choices=['resnet34', 'mobilenet_v2'],help = "Model to be used for evaluation, 'resnet34' OR 'mobilenet_v2")
parser.add_argument("weight_file", type=str, help="Path to the weight file")

args = parser.parse_args()


DATASET_PATH = args.dataset_path #Path of the dataset containing test images
WEIGHT_FILE_PATH = args.weight_file #path of the weight file of the model to be evaluated

device = get_default_device() #Obtain default evice: GPU or CPU

model_name = args.model
model = None
if model_name == 'resnet34':
  model = to_device(ASLResnet(), device) #Instantiate the model and transfer it to the available device
else:
  model = to_device(ASLMobilenet(), device)

  
model.load_state_dict(torch.load(WEIGHT_FILE_PATH)) #Load the weights onto the model

test_ds = ImageFolder(DATASET_PATH, transform = tt.Compose([tt.Scale((200,200)), tt.ToTensor()])) #Create a dataset object: Loads the images as tensors and resizes them to size 200x200
test_dl = DataLoader(test_ds, 16, num_workers=16, pin_memory=True) #Creates the dataloader object for the dataset
test_dl = DeviceDataLoader(test_dl, device) #Loads the dataloader onto the available device
model = model.eval() #Set the model to evaluation mode

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']


def create_confusion_matrix(y_true, y_pred, classes):
    """ creates and plots a confusion matrix given two list (targets and predictions)
    :param list y_true: list of all targets (in this case integers bc. they are indices)
    :param list y_pred: list of all predictions (in this case one-hot encoded)
    """
    classes = {}
    for l in labels:
      classes[l] = labels.index(l)
    amount_classes = len(classes)

    confusion_matrix = np.zeros((amount_classes, amount_classes)) #Initialise the confusion matrix with all zeros
    for idx in range(len(y_true)):
        target = y_true[idx]

        output = y_pred[idx]

        confusion_matrix[target][output] += 1 #Add 1 to the appropriate cell (target, predicted output)


    ##### PLOT THE MATRIX ######################################################
    plt.figure(figsize = (5,5))
    fig, ax = plt.subplots(1)

    ax.matshow(confusion_matrix)
    ax.set_xticks(np.arange(len(list(classes.keys()))))
    ax.set_yticks(np.arange(len(list(classes.keys()))))

    ax.set_xticklabels(list(classes.keys()))
    ax.set_yticklabels(list(classes.keys()))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    #plt.show()
    filename = 'confusion_matrix.png'
    if os.path.isfile(filename):
      os.remove(filename)
    plt.savefig(filename)
    ############################################################################

def find_metrics(model, test_dl):
  """
  Finds F1 score and plots the confusion matrix. Parameters: model and test dataloader.
  """
  predictions, targets = [], []
  for images, labels in test_dl: #Iterate through the dataloader
    logps = model(images) #Find output from model
    output = torch.exp(logps)
    pred = torch.argmax(output, 1) #Find maximum output for finding alphabet label

    # convert to numpy arrays
    pred = pred.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    for i in range(len(pred)): #Append the predictions and targets in a list
      predictions.append(pred[i])
      targets.append(labels[i])

  F1 = f1_score(targets, predictions, average = 'micro') #find F1 score through ithe inbuilt function of scikit-learn
  create_confusion_matrix(targets, predictions, labels) #plot the confusion matrix
  
  
  print('F1 score: ', F1) #print out the F1 score


find_metrics(model, test_dl) #call the function to calculate metrics and plot confusion matrix


